import argparse
import os
import random
import numpy as np
import sys
import evaluate
import pandas as pd
import torch
import logging
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from models import (
    ModelTextCNNForSequenceClassification,
    ModelRNNForSequenceClassification,
    ModelRCNNForSequenceClassification
)
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    Trainer, TrainingArguments)


logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('training')
file_handler = logging.FileHandler('train.log') 
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

sys.path.append('./')

_PARSER = argparse.ArgumentParser('ptm detector')
_PARSER.add_argument('--train_doc', type=str, default='../../data/zh_doc_train.csv', help='input train file path')
_PARSER.add_argument('--test_doc', type=str, default='../../data/zh_doc_test.csv', help='input doc test file path')
_PARSER.add_argument('--test_sent', type=str, default='../../data/shuffled_zh_sent_test.csv', help='input test sent file path')
_PARSER.add_argument('--model_name', type=str, default='hfl/chinese-roberta-wwm-ext', help='ptm model name')
_PARSER.add_argument('--checkpoint', type=str, default="checkpoint",help='location for saving models')
_PARSER.add_argument('--batch_size', type=int, default=16, help='batch size')
_PARSER.add_argument('--epochs', type=int, default=2, help='epochs')
_PARSER.add_argument('--lr', type=float, default=5e-5, help='learning rate')
_PARSER.add_argument('--weight_decay', type=float, default=0.01, help='weight_decay')
_PARSER.add_argument('--num_labels', type=int, default=2, help='num_labels')
_PARSER.add_argument('--cnn_kernel_size', type=int, default=256, help='cnn_kernel_size')
_PARSER.add_argument('--cnn_drop_rate', type=float, default=0.5, help='cnn_drop_rate')
_PARSER.add_argument('--cnn_lr', type=float, default=2e-4, help='cnn layer learning rate')
_PARSER.add_argument('--rnn_type', type=str, default="gru", help='type of rnn')
_PARSER.add_argument('--rnn_hidden_dim', type=int, default=256, help='rnn_hidden_dim')
_PARSER.add_argument('--rnn_nums_layer', type=int, default=2, help='rnn_nums_layer')
_PARSER.add_argument('--rnn_drop_rate', type=float, default=0.5, help='num_labels')
_PARSER.add_argument('--rnn_bidirectional', type=bool, default=True, help='rnn_bidirectional')
_PARSER.add_argument('--rnn_batch_first', type=bool, default=True, help='rnn_batch_first')
_PARSER.add_argument('--rnn_lr', type=float, default=2e-4, help='rnn layer learning rate')
_PARSER.add_argument('--cuda', type=str, default='0', help='gpu ids, like: 1,2,3')
_PARSER.add_argument('--seed', type=int, default=42, help='random seed.')
_PARSER.add_argument('--max_length', type=int, default=365, help='max_length')
_PARSER.add_argument('--use_cnn', default=False, help='use_cnn_model')
_PARSER.add_argument('--use_rnn', default=False, help='use_rnn_model')
_PARSER.add_argument('--use_rcnn', default=False, help='use_rcnn_model')

_ARGS = _PARSER.parse_args()

if len(_ARGS.cuda) > 1:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

os.environ["OMP_NUM_THREADS"] = '8'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # if cuda >= 10.2
os.environ['CUDA_VISIBLE_DEVICES'] = _ARGS.cuda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataloader(args: argparse.Namespace):
    """
    dataloaders分别是train_doc, test_doc, test_sent
    """
    datasets = []
    files = [args.train_doc, args.test_doc, args.test_sent]
    for file in files:
        df = pd.read_csv(file)
        dataset = Dataset.from_pandas(df)
        datasets.append(dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    def tokenize_fn(example):
        return tokenizer(example['answer'], max_length=args.max_length, padding='max_length', truncation=True)
    datasets = [datasets[0], datasets[1], datasets[2]]
    names = ['id', 'question', 'answer', 'source']
    tokenized_datasets = []
    for dataset in datasets:
        tokenized = dataset.map(
                        tokenize_fn,
                        batched=True,
                        remove_columns=names)
        tokenized_datasets.append(tokenized)
    def collate_fn(examples):
        return tokenizer.pad(examples,return_tensors='pt')
    
    dataloaders = []
    for dataset in tokenized_datasets:
        # dataloader = DataLoader(dataset, shuffle=(dataset != tokenized_datasets[0]), collate_fn=collate_fn, batch_size=args.batch_size)
        dataloader = DataLoader(dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size)
        dataloaders.append(dataloader)
    return dataloaders

def get_model_and_optimizer(args):
    if args.use_cnn:
        logger.info("***** Running training %s CNN*****", args.model_name.split('/')[-1])
        output_dir = os.path.join(args.checkpoint, "RoBERTa_CNN")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model = ModelTextCNNForSequenceClassification(args)
        optimizer = AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        bert_parameters = list(model.model.parameters()) 
        cnn_parameters = list(model.convs.parameters()) 
        classifier_parameters = list(model.classifier.parameters())
        optimizer = torch.optim.AdamW([
            {'params': bert_parameters, 'lr': args.lr},
            {'params': cnn_parameters, 'lr': args.cnn_lr},
            {'params': classifier_parameters, 'lr': args.lr}
        ], weight_decay=args.weight_decay)
    elif args.use_rnn:
        logger.info("***** Running training %s RNN*****", args.model_name.split('/')[-1])
        output_dir = os.path.join(args.checkpoint, "RoBERTa_RNN")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model = ModelRNNForSequenceClassification(args)
        bert_parameters = list(model.model.parameters()) 
        rnn_parameters = list(model.rnn.parameters()) 
        classifier_parameters = list(model.classifier.parameters())
        optimizer = torch.optim.AdamW([
            {'params': bert_parameters, 'lr': args.lr},
            {'params': rnn_parameters, 'lr': args.rnn_lr},
            {'params': classifier_parameters, 'lr': args.lr}
        ], weight_decay=args.weight_decay)
    elif args.use_rcnn:
        logger.info("***** Running training %s RCNN*****", args.model_name.split('/')[-1])
        output_dir = os.path.join(args.checkpoint, "RoBERTa_RCNN")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model = ModelRCNNForSequenceClassification(args)
        bert_parameters = list(model.model.parameters()) 
        rnn_parameters = list(model.rnn.parameters()) 
        classifier_parameters = list(model.classifier.parameters())
        optimizer = torch.optim.AdamW([
            {'params': bert_parameters, 'lr': args.lr},
            {'params': rnn_parameters, 'lr': args.rnn_lr},
            {'params': classifier_parameters, 'lr': args.lr}
        ], weight_decay=args.weight_decay)
    else:
        logger.info("***** Running training %s *****", args.model_name.split('/')[-1])
        output_dir = os.path.join(args.checkpoint, args.model_name.split('/')[-1])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels, trust_remote_code=True)
        optimizer = AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer, output_dir


def main(args: argparse.Namespace, dataloaders):
    """
    确保每次运行结果一致
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    """
    根据模型不同设置
    """
    model, optimizer, output_dir = get_model_and_optimizer(args)
    model.to(device)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(dataloaders[0]) * args.epochs),
        num_training_steps=(len(dataloaders[0]) * args.epochs),
    )

    total_step = 0 
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(tqdm(dataloaders[0], desc='Training', colour="green")):
            batch.to(device)
            labels = batch.pop('label')
            outputs = model(**batch)
            loss = F.cross_entropy(outputs['logits'], labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_step += 1
            if total_step % 550 == 0 and total_step != 0:
                torch.save(model, f"{output_dir}/{total_step}_step_model")
                """
                评估模型
                """
                model.eval()
                eval_name_list = ['test_doc', 'test_sent']
                for item, eval_name in enumerate(eval_name_list, 1):
                    metric = evaluate.load("f1", trust_remote_code=True)
                    for step, batch in enumerate(tqdm(dataloaders[item], desc='Evaling', colour="green")):
                        batch.to(device)
                        with torch.no_grad():
                            labels = batch.pop('label')
                            outputs = model(**batch)
                        predictions = outputs['logits'].argmax(dim=-1)
                        predictions, references = predictions, labels
                        metric.add_batch(
                            predictions=predictions,
                            references=references,
                        )
                    eval_metric = metric.compute()
                    logger.info(f"{eval_name} epoch {epoch}: {eval_metric}")
                """
                eval之后恢复训练模式
                """
                model.train()
    logger.info("***** Finish training *****")
if __name__== '__main__':
    dataloaders = create_dataloader(_ARGS)
    main(_ARGS, dataloaders)