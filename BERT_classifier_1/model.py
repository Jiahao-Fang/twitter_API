# coding: utf-8
# @File: model.py
# @Author: Jiahao
# @Time: 2020/11/05 17:12:56
# @Description:

import torch
import torch.nn as nn
from transformers import  BertModel

# Bert
class BertClassifier(nn.Module):
    def __init__(self, bert_config, num_labels):
        super().__init__()
        # Define BERT model
        self.bert = BertModel(config=bert_config)
        # Define Classifier
        self.classifier = nn.Linear(bert_config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids): # forward method, (the method we dealt with the tersor after encoding)
        # Output of bert
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # the pooled output of [CLS]
        pooled = bert_output[1]
        # Classification
        logits = self.classifier(pooled)
        # softmax
        return torch.softmax(logits, dim=1)

# Bert+BiLSTM
##Could be called in the train.py
class BertLstmClassifier(nn.Module):
    def __init__(self, bert_config):
        super().__init__()
        self.bert = BertModel(config=bert_config)
        self.lstm = nn.LSTM(input_size=bert_config.hidden_size, hidden_size=bert_config.hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(bert_config.hidden_size*2, bert_config.num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output, pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out, _ = self.lstm(output)
        logits = self.classifier(out[:, -1, :])
        return self.softmax(logits)
