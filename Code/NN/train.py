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
    FastText,
    TextCNN,
    RNNModel
)
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pickle as pkl
from transformers import (
    get_linear_schedule_with_warmup)


logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('training')
file_handler = logging.FileHandler('train.log') 
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

sys.path.append('./')

_PARSER = argparse.ArgumentParser('nn detector')
_PARSER.add_argument('--train_doc', type=str, default='doc_train.txt', help='input train file path')
_PARSER.add_argument('--test_doc', type=str, default='doc_test.txt', help='input doc test file path')
_PARSER.add_argument('--test_sent', type=str, default='sent_test.txt', help='input test sent file path')
_PARSER.add_argument('--model_name', type=str, default='RNNModel', help='nn model name')
_PARSER.add_argument('--embedding_file', type=str, default='embedding_zh.npz', help='embedding file path')

_PARSER.add_argument('--embedding_dim', type=int, default=300, help='embedding_size')
_PARSER.add_argument('--checkpoint', type=str, default="checkpoint",help='location for saving models')
_PARSER.add_argument('--batch_size', type=int, default=16, help='batch size')
_PARSER.add_argument('--epochs', type=int, default=15, help='epochs')
_PARSER.add_argument('--lr', type=float, default=2e-4, help='learning rate')
_PARSER.add_argument('--weight_decay', type=float, default=0.01, help='weight_decay')
_PARSER.add_argument('--num_labels', type=int, default=2, help='num_labels')

_PARSER.add_argument('--FastText_drop_rate', type=float, default=0.5, help='FastText_drop_rate')
_PARSER.add_argument('--cnn_kernel_size', type=int, default=256, help='cnn_kernel_size')
_PARSER.add_argument('--cnn_drop_rate', type=float, default=0.5, help='cnn_drop_rate')

_PARSER.add_argument('--rnn_type', type=str, default="gru", help='type of rnn')
_PARSER.add_argument('--rnn_hidden_dim', type=int, default=256, help='rnn_hidden_dim')
_PARSER.add_argument('--rnn_nums_layer', type=int, default=2, help='rnn_nums_layer')
_PARSER.add_argument('--rnn_drop_rate', type=float, default=0.5, help='num_labels')
_PARSER.add_argument('--rnn_bidirectional', type=bool, default=True, help='rnn_bidirectional')
_PARSER.add_argument('--rnn_batch_first', type=bool, default=True, help='rnn_batch_first')

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


word_to_id = pkl.load(open("vocab.pkl", 'rb'))  #加载词典
def tokenize_fun(s):  # 输入句子s
    max_size = 365# 句子分词最大长度
    ts = [w for i, w in enumerate(s) if i < max_size]  # 得到字符列表，最多32个
    ids = [word_to_id[w] if w in word_to_id.keys() else word_to_id['[UNK]'] for w in ts]  # 根据词典，将字符列表转换为id列表
    ids += [0 for _ in range(max_size-len(ts))]  # 若id列表达不到最大长度，则补0
    return ids

class MyData(Dataset):  # 继承Dataset
    def __init__(self, tokenize_fun, filename):
        self.filename = filename  # 要加载的数据文件名
        self.tokenize_function = tokenize_fun  # 实例化时需传入分词器函数
        print("Loading dataset "+ self.filename +" ...")
        self.data, self.labels = self.load_data()  # 得到分词后的id序列和标签
    #读取文件，得到分词后的id序列和标签，返回的都是tensor类型的数据
    def load_data(self):
        labels, data = [], []
        with open(self.filename, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc='Loading data', colour="green"):
                fields  = line.strip().split('\t')
                if len(fields) != 2:
                    continue
                labels.append(float(fields[1]))  # one-ho
                data.append(self.tokenize_function(fields[0]))
        f.close()
        return torch.tensor(data), torch.tensor(labels)
    def __len__(self):  # 返回整个数据集的大小
        return len(self.data)
    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        return self.data[index], self.labels[index]

def getDataLoader(doc_train_dataset, doc_test_dataset, sent_test_dataset):
    batch_size = 16
    dataloaders = []
    doc_train_dataloader = DataLoader(
        dataset=doc_train_dataset,
        batch_size=batch_size,  # 从数据集合中每次抽出batch_size个样本
        shuffle=True,  # 加载数据时打乱样本顺序
    )
    dataloaders.append(doc_train_dataloader)
    doc_test_dataloader = DataLoader(
        dataset=doc_test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    dataloaders.append(doc_test_dataloader)
    sent_test_dataloader = DataLoader(
        dataset=sent_test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 按原始数据集样本顺序加载
    )
    dataloaders.append(sent_test_dataloader)
    return dataloaders

def train(args, dataloaders):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.model_name == "FastText":
        model = FastText(args)
    elif args.model_name == "TextCNN":
        model = TextCNN(args)
    else:
        model = RNNModel(args)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(dataloaders[0]) * args.epochs),
        num_training_steps=(len(dataloaders[0]) * args.epochs),
    )

    output_dir = os.path.join(args.checkpoint, args.model_name.split('/')[-1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for epoch in range(args.epochs):
        model.train()
        for step, (trains, labels) in enumerate(tqdm(dataloaders[0], desc='Training', colour="green")):
            outputs = model(trains.to(device))
            loss = F.cross_entropy(outputs['logits'], labels.long().to(device))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        torch.save(model, f"{output_dir}/The epoch{epoch}_model")
        """
        评估模型
        """
        model.eval()
        eval_name_list = ['test_doc', 'test_sent']
        for item, eval_name in enumerate(eval_name_list, 1):
            metric = evaluate.load("f1", trust_remote_code=True)
            for step, (trains, labels) in enumerate(tqdm(dataloaders[item], desc='Evaling', colour="green")):
                with torch.no_grad():
                    outputs = model(trains.to(device))
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
    args = _ARGS
    train_doc_dataset = MyData(tokenize_fun=tokenize_fun, filename=args.train_doc)
    test_doc_dataset = MyData(tokenize_fun=tokenize_fun, filename=args.test_doc)
    test_sent_dataset = MyData(tokenize_fun=tokenize_fun, filename=args.test_sent)
    dataloaders = getDataLoader(train_doc_dataset, test_doc_dataset, test_sent_dataset)
    train(args, dataloaders)