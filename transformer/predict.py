# -*-coding:utf-8-*-
# ---
# @Fila: predict.py
# @Author: Jiahao Fang
# @Date: 2022/11/8
# @Description:
# @Project: sentiment_analysis

import time
import torch
import numpy as np
from importlib import import_module

dataset = 'cointelegraph'  # 数据集
embedding = 'random'
word = True
mode = 'test'  #不进行训练，只测试
model_name = 'Bert'
from utils import build_dataset, build_iterator, get_time_dif

x = import_module('models.' + model_name)
config = x.Config(dataset, embedding)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

start_time = time.time()
print("Loading data...")
test_data,test_raws = build_dataset(config, word, mode)
test_iter = build_iterator(test_data, config)
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

model = x.Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path))
model.eval()
start_time = time.time()
model.eval()
loss_total = 0
predict_all = np.array([], dtype=int)
labels_all = np.array([], dtype=int)
with torch.no_grad():
    for texts, labels in test_iter:
        outputs = model(texts)
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, labels.cpu().numpy())

for label,content in zip(labels_all.tolist(),test_raws):
    print(str(label) + "\t" + content)