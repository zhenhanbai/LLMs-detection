import argparse
import os
import numpy as np
import sys
import evaluate
import pandas as pd
import torch
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer)

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('testing')
file_handler = logging.FileHandler('test.log') 
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

sys.path.append('./')

_PARSER = argparse.ArgumentParser('ptm detector')
_PARSER.add_argument('--bert_model', type=str, default='checkpoint/bert-base-chinese/2200_step_model', help='bert_model')
_PARSER.add_argument('--roberta_model',type=str, default='checkpoint/chinese-roberta-wwm-ext/2200_step_model', help='roberta_model')
_PARSER.add_argument('--xlnet_model',type=str, default='checkpoint/chinese-xlnet-base/2200_step_model', help='xlnet_model')
_PARSER.add_argument('--ernie_model',type=str, default='checkpoint/ernie-3.0-base-zh/2200_step_model', help='ernie_model')
_PARSER.add_argument('--roberta_cnn_model',type=str, default='checkpoint/RoBERTa_CNN/2200_step_model', help='roberta_cnn_model')
_PARSER.add_argument('--roberta_rnn_model',type=str, default='checkpoint/RoBERTa_RNN/2200_step_model', help='roberta_rnn_model')
_PARSER.add_argument('--roberta_rcnn_model',type=str, default='checkpoint/RoBERTa_RCNN/2200_step_model', help='roberta_rcnn_model')
_PARSER.add_argument('--test_doc', type=str, default='../../data/zh_doc_test.csv', help='input doc test file path')
_PARSER.add_argument('--test_sent', type=str, default='../../data/shuffled_zh_sent_test.csv', help='input test sent file path')
_PARSER.add_argument('--batch_size', type=int, default=16, help='batch size')
_PARSER.add_argument('--model_name', type=str, default='hfl/chinese-roberta-wwm-ext', help='ptm model name')
_PARSER.add_argument('--epochs', type=int, default=2, help='epochs')
_PARSER.add_argument('--num_labels', type=int, default=2, help='num_labels')
_PARSER.add_argument('--cuda', type=str, default='0', help='gpu ids, like: 1,2,3')
_PARSER.add_argument('--seed', type=int, default=42, help='random seed.')
_PARSER.add_argument('--max_length', type=int, default=365, help='max_length')
_PARSER.add_argument('--stacking', type=bool, default=True, help='stacking')

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
    files = [args.test_doc, args.test_sent]
    for file in files:
        df = pd.read_csv(file)
        dataset = Dataset.from_pandas(df)
        datasets.append(dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    def tokenize_fn(example):
        return tokenizer(example['answer'], max_length=args.max_length, padding='max_length', truncation=True)
    datasets = [datasets[0], datasets[1]]
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
        dataloader = DataLoader(dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)
        dataloaders.append(dataloader)
    return dataloaders

def eval(args, dataloaders):
    if args.stacking:
        roberta_cnn_model = torch.load(args.roberta_cnn_model).to(device)
        roberta_cnn_model.eval()
        print("roberta_cnn_model loaded")
        roberta_model = torch.load(args.roberta_model).to(device)
        roberta_model.eval()
        print("roberta_rnn_model loaded")
        roberta_rcnn_model = torch.load(args.roberta_rcnn_model).to(device)
        roberta_rcnn_model.eval()
        print("roberta_rcnn_model loaded")
        eval_name_list = ['test_doc', 'test_sent']
        for item, eval_name in enumerate(eval_name_list, 0):
            metric = evaluate.load("f1", trust_remote_code=True)
            for step, batch in enumerate(tqdm(dataloaders[item], desc='Evaling', colour="green")):
                batch.to(device)
                with torch.no_grad():
                    labels = batch.pop('label')
                    outputs = (roberta_cnn_model(**batch)['logits'] * 0.6 + roberta_rcnn_model(**batch)['logits'] * 0.3 + roberta_model(**batch)['logits'] * 0.1)
                predictions = outputs.argmax(dim=-1)
                predictions, references = predictions, labels
                metric.add_batch(
                    predictions=predictions,
                    references=references,
                )
            eval_metric = metric.compute()
            logger.info(f"{eval_name}: {eval_metric}")

daataLoader = create_dataloader(_ARGS)
eval(_ARGS,daataLoader)