# coding: utf-8
# @File: dataset.py
# @Author: Jiahao
# @Time: 2022/11/06 11:01:32
# @Description:

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm
import time
class CNewsDataset(Dataset):
    def __init__(self, filename):
        # Initialize the dataset
        self.labels = [0,1,2]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base')
        self.input_ids = [] ## id of the word
        self.token_type_ids = []## the id of the sentenses
        self.attention_mask = [] ## padding
        self.label_id = []
        self.load_data(filename)
    
    def load_data(self, filename):
        # load data
        print('loading data from:', filename)
        with open(filename, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
        for line in tqdm(lines, ncols=50):
            text,label = line.strip().split('\t')
            ## tokenlize the text
            token = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=512)
            self.input_ids.append(np.array(token['input_ids']))
            self.token_type_ids.append(np.array(token['token_type_ids']))
            self.attention_mask.append(np.array(token['attention_mask']))
            self.label_id.append(int(label))

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.label_id[index]

    def __len__(self):
        return len(self.input_ids)

