from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np


class ModelTextCNNForSequenceClassification(nn.Module):
    def __init__(self, args):
        super(ModelTextCNNForSequenceClassification, self).__init__()
        self.model = AutoModel.from_pretrained(args.model_name)
        if args.model_name != "hfl/chinese-xlnet-base":
            self.convs = nn.ModuleList(
                [nn.Conv2d(1, args.cnn_kernel_size, (k, self.model.config.hidden_size)) for k in [2, 3, 4]])
        else:
            self.convs = nn.ModuleList(
                [nn.Conv2d(1, args.cnn_kernel_size, (k, self.model.config.d_model)) for k in [2, 3, 4]])
        self.dropout_cnn = nn.Dropout(args.cnn_drop_rate)
        self.classifier = nn.Linear(args.cnn_kernel_size * 3, args.num_labels)

    def conv_and_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2) 
        return x
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None):

        outputs = self.model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        
        seq_output = outputs[0]
        out = seq_output.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout_cnn(out)
        logits = self.classifier(out)    
        return {'logits':logits}
    
class ModelRNNForSequenceClassification(nn.Module):
    def __init__(self, args):
        super(ModelRNNForSequenceClassification, self).__init__()
        self.model = AutoModel.from_pretrained(args.model_name)
        self.rnn_type = args.rnn_type
        if args.model_name != "hfl/chinese-xlnet-base":
            self.embedding_size = self.model.config.hidden_size
        else:
            self.embedding_size = self.model.config.d_model
        if args.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size,
                               hidden_size=args.rnn_hidden_dim,
                               num_layers=args.rnn_nums_layer,
                               bidirectional=args.rnn_bidirectional,
                               batch_first=args.rnn_batch_first,
                               dropout=args.rnn_drop_rate)
        elif args.rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size,
                               hidden_size=args.rnn_hidden_dim,
                               num_layers=args.rnn_nums_layer,
                               bidirectional=args.rnn_bidirectional,
                               batch_first=args.rnn_batch_first,
                               dropout=args.rnn_drop_rate)
        else:
            self.rnn = nn.RNN(self.embedding_size,
                                hidden_size=args.rnn_hidden_dim,
                                num_layers=args.rnn_nums_layer,
                                bidirectional=args.rnn_bidirectional,
                                batch_first=args.rnn_batch_first,
                                dropout=args.rnn_drop_rate)
            
        self.dropout= nn.Dropout(p=args.rnn_drop_rate)

        if args.rnn_bidirectional:
            self.classifier = nn.Linear(args.rnn_hidden_dim * 2, args.num_labels)
        else:
            self.classifier = nn.Linear(args.rnn_hidden_dim, args.num_labels)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None):

        outputs = self.model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        seq_output = outputs[0]
        self.rnn.flatten_parameters() # 扁平化
        if self.rnn_type in ['rnn', 'gru']:
            output, hidden = self.rnn(seq_output)
        else:
            output, (hidden, cell) = self.rnn(seq_output)
        x = output[:, -1, :]
        x = self.dropout(x)
        logits = self.classifier(x)

        return {"logits":logits}
    

class ModelRCNNForSequenceClassification(nn.Module):
    def __init__(self, args):
        super(ModelRCNNForSequenceClassification, self).__init__()
        self.model = AutoModel.from_pretrained(args.model_name)
        self.rnn_type = args.rnn_type
        if args.model_name != "hfl/chinese-xlnet-base":
            self.embedding_size = self.model.config.hidden_size
        else:
            self.embedding_size = self.model.config.d_model
        if args.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size,
                                hidden_size=args.rnn_hidden_dim,
                                num_layers=args.rnn_nums_layer,
                                bidirectional=args.rnn_bidirectional,
                                batch_first=args.rnn_batch_first,
                                dropout=args.rnn_drop_rate)
        elif args.rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size,
                                hidden_size=args.rnn_hidden_dim,
                                num_layers=args.rnn_nums_layer,
                                bidirectional=args.rnn_bidirectional,
                                batch_first=args.rnn_batch_first,
                                dropout=args.rnn_drop_rate)
        else:
            self.rnn = nn.RNN(self.embedding_size,
                                hidden_size=args.rnn_hidden_dim,
                                num_layers=args.rnn_nums_layer,
                                bidirectional=args.rnn_bidirectional,
                                batch_first=args.rnn_batch_first,
                                dropout=args.rnn_drop_rate)
            
        self.dropout= nn.Dropout(p=args.rnn_drop_rate)
        self.maxpool = nn.MaxPool1d(args.max_length)
        self.ReLU = nn.ReLU()

        if args.rnn_bidirectional:
            self.classifier = nn.Linear(args.rnn_hidden_dim * 2 + self.embedding_size, args.num_labels)
        else:
            self.classifier = nn.Linear(args.rnn_hidden_dim + self.embedding_size, args.num_labels)

    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None):

        outputs = self.model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        seq_output = outputs[0]
        self.rnn.flatten_parameters() # 扁平化
        if self.rnn_type in ['rnn', 'gru']:
            output, hidden = self.rnn(seq_output)
        else:
            output, (hidden, cell) = self.rnn(seq_output)
        x = torch.cat([seq_output, output], dim=2) # 连接Bertmodel 和 rnnmodel 的输出
        x = self.ReLU(x) # 非线性变化
        x = x.permute(0, 2, 1) # [batch_size, embedding_dim, max_seq]
        x = self.maxpool(x).squeeze(2) # [batch_size, embedding_dim]
        x = self.dropout(x)
        logits = self.classifier(x)

        return {"logits":logits}