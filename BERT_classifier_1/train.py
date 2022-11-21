# coding: utf-8
# @File: train.py
# @Author: Jiahao
# @Time: 2022/11/18 17:14:07
# @Description:

import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW, BertConfig
from torch.utils.data import DataLoader
from model import BertClassifier
from dataset import CNewsDataset
from tqdm import tqdm

def main():

    # Parameter setup
    batch_size = 8
    # If device is GPU supported
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10
    # learning_rate 3e-6 to 1e-5
    learning_rate = 5e-6

    # Obtain dataset
    train_dataset = CNewsDataset('./cointelegraph/data/train.txt')
    valid_dataset = CNewsDataset('./cointelegraph/data/dev.txt')


    # generate the batch by DataLoader
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # read the config of bert
    bert_config = BertConfig.from_pretrained('bert-base')
    num_labels = len(train_dataset.labels)
    # Initialize the Bert model
    model = BertClassifier(bert_config, 3).to(device)

    # weight decay on Optimizer
    ## Copied from online, people uses this decay in the BERT, the decaying learning rate will improve the model performance
    ## according to the intepretation 
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] ## The no decay parameter
    optimizer_grouped_parameters = [
        ## parameter learning rate decay to 0.99 each time
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01}, 
        ## parameter no decay
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters,lr=learning_rate)
    # Loss function
    criterion = nn.CrossEntropyLoss()

    best_acc = 0

    for epoch in range(1, epochs+1):
        losses = 0      # Loss
        accuracy = 0    # Accuracy

        model.train() ## pytorch train function
        ## initialize the batch from train data
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_bar = tqdm(train_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id in train_bar:
            train_bar.set_description('Epoch %i train' % epoch)
            # Process data to device with long type,
            # pytorch will call model.forward to output the classification of the datepoints
            output = model(
                input_ids=input_ids.long().to(device),
                attention_mask=attention_mask.long().to(device),
                token_type_ids=token_type_ids.long().to(device),
            )##output is a batch*3 vector 
            # clear gradient
            model.zero_grad()
            # loss calculation
            loss = criterion(output, label_id.to(device))
            losses += loss.item()
            ## predict the label by argmax
            pred_labels = torch.argmax(output, dim=1)
            ## accuracy calculation
            acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels) #acc
            accuracy += acc

            # back propagation of the parameter
            loss.backward()
            # update the gradient
            optimizer.step()
            train_bar.set_postfix(loss=loss.item(), acc=acc)


        average_loss = losses / len(train_dataloader)
        average_acc = accuracy / len(train_dataloader)

        print('\tTrain ACC:', average_acc, '\tLoss:', average_loss)

        # validation (Similar to training)
        model.eval()
        losses = 0   
        accuracy = 0 
        valid_bar = tqdm(valid_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id  in valid_bar:
            valid_bar.set_description('Epoch %i valid' % epoch)

            output = model(
                input_ids=input_ids.long().to(device),
                attention_mask=attention_mask.long().to(device),
                token_type_ids=token_type_ids.long().to(device),
            )
            
            loss = criterion(output, label_id.to(device))
            losses += loss.item()

            pred_labels = torch.argmax(output, dim=1)
            acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)
            accuracy += acc
            valid_bar.set_postfix(loss=loss.item(), acc=acc)

        average_loss = losses / len(valid_dataloader)
        average_acc = accuracy / len(valid_dataloader)

        print('\tValid ACC:', average_acc, '\tLoss:', average_loss)

        if not os.path.exists('models'):
            os.makedirs('models')
        
        # save the best performance model
        if average_acc > best_acc:
            best_acc = average_acc
            torch.save(model.state_dict(), 'models/best_model.pkl')
        
if __name__ == '__main__':
    main()