import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import random
from collections import defaultdict
import os
import h5py
import sys, os
from torch.utils.data import DataLoader
import time
import math
from torch.nn import utils
import re

from data_helper import helper
from multi_task import MultiTask
from hyperboard import Agent
from folder import Folder
from evaluate import evaluate

from transform import Transform
transform = Transform("/data/xuwenshen/workspace/squad/data/train_dev_voc.json")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='which gpu to run', default = '0')
parser.add_argument('--port', help='hyperboard port', type=int, default = 5000)
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


hidden_size = 156
dropout = 0.5
passage_len = 350
lr = 0.001
batch_size = 24
epochs = 1000
lambda_m = 0
lambda_g = 0
lambda_c = 1


voc_path = "/data/xuwenshen/workspace/squad/data/train_dev_voc.json"
embedding_path = "/data/xuwenshen/workspace/squad/data/train_dev_embedding.json"

net = MultiTask(hidden_size=hidden_size, dropout=dropout, passage_len=passage_len,voc_path=voc_path, embedding_path=embedding_path)
print (net)
if torch.cuda.is_available():
    net.cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = lr)
net.train()

hyperparameters = defaultdict(lambda:0)
hyperparameters['criteria'] = None
hyperparameters['dropout'] = dropout
hyperparameters['lr'] = lr
hyperparameters['batch_size'] = batch_size
hyperparameters['hidden_size'] = hidden_size
hyperparameters['lambda_m'] = lambda_m
hyperparameters['lambda_g'] = lambda_g
hyperparameters['lambda_c'] = lambda_c


agent = Agent(port=args.port)

hyperparameters['criteria'] = 'train match loss'
train_match_loss = agent.register(hyperparameters, 'loss')

hyperparameters['criteria'] = 'valid match loss'
valid_match_loss = agent.register(hyperparameters, 'loss')

hyperparameters['criteria'] = 'valid match em'
valid_match_em = agent.register(hyperparameters, 'em')

hyperparameters['criteria'] = 'valid match f1'
valid_match_f1 = agent.register(hyperparameters, 'f1')

hyperparameters['criteria'] = 'train generation loss'
train_generation_loss = agent.register(hyperparameters, 'loss')


hyperparameters['criteria'] = 'train classification loss'
train_classification_loss = agent.register(hyperparameters, 'loss')

hyperparameters['criteria'] = 'train loss'
train_loss = agent.register(hyperparameters, 'loss')

train_folder = Folder(filepath='/data/xuwenshen/workspace/squad/data/train.json', number_data = 1000, voc_path=voc_path)
train_loader = DataLoader(train_folder, batch_size=batch_size, num_workers=1, shuffle=True)

valid_folder = Folder(filepath='/data/xuwenshen/workspace/squad/data/dev.json', number_data = 100, voc_path=voc_path)
valid_loader = DataLoader(valid_folder, batch_size=batch_size, num_workers=1, shuffle=True)


def prepare_classify(question_tokens, answer_tokens):
    answer_index = list(range(question_tokens.size(0)))
    for i in range(0, int(question_tokens.size(0)/2)):
        sample = random.randint(0, question_tokens.size(0)-1)
        answer_index[sample] = random.randint(0, question_tokens.size(0)-1)
    answer_index = np.array(answer_index)
    labels = torch.Tensor((np.array(range(question_tokens.size(0))) == answer_index).tolist())
    return labels.long(), torch.from_numpy(answer_index).long()

def valid():
    
    net.eval()
    dev_loss = 0 
    batch = 0
    
    hypothesis = defaultdict(lambda:0)
    answers = defaultdict(lambda:0)
    
    to_print = True
    for tdata in valid_loader:
        
        passage_tokens = tdata['passage_tokens']
        passage_len = tdata['passage_len']
        char_start_end = tdata['char_start_end'] 
        question_tokens = tdata['question_tokens'] 
        question_len = tdata['question_len']
        ground_truth = tdata['ground_truth']
        answer_tokens = tdata['answer_tokens']
        answer_len = tdata['answer_len']
        boundary = tdata['boundary']
        passage_str = tdata['passage_str']
        question_str = tdata['question_str']
        answer_str = tdata['answer_str']
        key = tdata['key']
        
        fw_res = net(passage_tokens, question_tokens)
        match_logits = fw_res['match_logits']
        match_predictions = fw_res['match_predictions']
        
        loss = net.get_loss(match_logits=match_logits, match_labels=boundary)
        match_loss = loss['match_loss']
        
        for i in range(len(match_predictions)):
            start = match_predictions[i][0]
            end = match_predictions[i][1]
            
            str_ = passage_str[i][char_start_end[i][start][0]:char_start_end[i][end][1]]
            
            hypothesis[key[i]] = str_
            
            if to_print:
                print (passage_str[i])
                print ('--------------------------')
                print (question_str[i])
                print ('==========================')
                print (answer_str[i], ' | ', str_)
                print ('**************************')
                print ('\n')
        to_print = False
    
        dev_loss += match_loss
        batch += 1
        
        del match_loss,fw_res, match_logits, match_predictions, loss
        
      
    dev_loss /= batch
    _ = evaluate(hypothesis)
    em = _['exact_match']
    f1 = _['f1']
    
    return dev_loss, em, f1

def save_model(net, dev_loss, em, f1, global_steps):
    
    model_dir = '/data/xuwenshen/workspace/squad/code/multi_task/models/'
    
    model_dir = model_dir + "loss-{:3f}-em-{:3f}-f1-{:3f}-steps-{:d}-model.pkl".format(dev_loss, em, f1, global_steps)
    
    torch.save(net.state_dict(), model_dir)

    
def check(net, tdata):
    
    net.eval()
    
    passage_tokens = tdata['passage_tokens']
    passage_len = tdata['passage_len']
    char_start_end = tdata['char_start_end'] 
    question_tokens = tdata['question_tokens'] 
    question_len = tdata['question_len']
    ground_truth = tdata['ground_truth']
    answer_tokens = tdata['answer_tokens']
    answer_len = tdata['answer_len']
    boundary = tdata['boundary']
    passage_str = tdata['passage_str']
    question_str = tdata['question_str']
    answer_str = tdata['answer_str']
    key = tdata['key']
    
    
    fw_res = net(passage=passage_tokens, 
                 question=question_tokens,
                 answer=answer_tokens,
                 decoder_inputs=ground_truth,
                 is_generation = True,
                 is_teacher_forcing = True)
        
    match_logits = fw_res['match_logits']
    match_predictions = fw_res['match_predictions']
    generation_predictions = fw_res['generation_predictions']
    
    
    loss_res = net.get_loss(match_logits=match_logits, match_labels=boundary)
            
    match_loss = loss_res['match_loss']
    loss = loss_res['loss']
    
    for i in range(len(match_predictions)):
        start = match_predictions[i][0]
        end = match_predictions[i][1]

        str_match = passage_str[i][char_start_end[i][start][0]:char_start_end[i][end][1]]
        
        str_generation = transform.i2t(generation_predictions[i])
        
        print (passage_str[i])
        print ('--------------------------')
        print (question_str[i])
        print ('--------------------------')
        print (str_generation)
        print ('==========================')
        print (answer_str[i], ' | ', str_match)
        print ('**************************')
        print ('\n')

    loss_value = 0 #sum(loss.cpu().data.numpy())
    del match_logits, match_predictions, loss, fw_res, generation_predictions
    
    return match_loss, loss_value
        
def train():
    
    global_steps = 0
    best_em = -1
    best_f1 = -1
    for iepoch in range(epochs):

        batch = 0
        for tdata in train_loader:

            passage_tokens = tdata['passage_tokens']
            passage_len = tdata['passage_len']
            char_start_end = tdata['char_start_end'] 
            question_tokens = tdata['question_tokens'] 
            question_len = tdata['question_len']
            ground_truth = tdata['ground_truth']
            answer_tokens = tdata['answer_tokens']
            answer_len = tdata['answer_len']
            boundary = tdata['boundary']
            passage_str = tdata['passage_str']
            question_str = tdata['question_str']
            answer_str = tdata['answer_str']
            key = tdata['key']

            classification_labels, answer_index = prepare_classify(question_tokens, answer_tokens)
            
            fw_res = net(passage=passage_tokens, 
                         question=question_tokens,
                         answer=answer_tokens,
                         answer_index=answer_index,
                         decoder_inputs=ground_truth, 
                         is_classification = True,
                         is_generation = True,
                         is_teacher_forcing = True)
            
            match_logits = fw_res['match_logits']
            match_predictions = fw_res['match_predictions']
            
            generation_logits = fw_res['generation_logits']
            generation_predictions = fw_res['generation_predictions']
            
            classification_logits = fw_res['classification_logits']
            classification_predictions = fw_res['classification_predictions']
            
            print (classification_predictions)
            print (classification_labels.numpy())
            print (sum(classification_labels.numpy() == classification_predictions) / len(classification_predictions))
            
            loss_return = net.get_loss(match_logits=match_logits, 
                                       match_labels=boundary, 
                                       generation_logits=generation_logits, 
                                       generation_labels=question_tokens, 
                                       classification_logits=classification_logits, 
                                       classification_labels= classification_labels,
                                       is_match = False,
                                       is_generation = False,
                                       is_classification = True,
                                       lambda_m = lambda_m,
                                       lambda_g = lambda_g,
                                       lambda_c = lambda_c)
            
            match_loss = loss_return['match_loss']
            loss = loss_return['loss']
            generation_loss = loss_return['generation_loss']
            classification_loss = loss_return['classification_loss']
            
            optimizer.zero_grad()
            loss.backward()
            utils.clip_grad_norm(net.parameters(), 5)
            optimizer.step()

            print (global_steps, iepoch, batch, 'match loss: ', match_loss, 'generation loss: ', generation_loss,'classification loss: ', classification_loss ) 
            agent.append(train_match_loss, global_steps, match_loss)
            agent.append(train_generation_loss, global_steps, generation_loss)
            agent.append(train_classification_loss, global_steps, classification_loss)
            agent.append(train_loss, global_steps, sum(loss.cpu().data.numpy()))
            
            batch += 1
            global_steps += 1
            del fw_res, match_logits, match_predictions, loss, match_loss, generation_loss, loss_return

            '''
            if global_steps % 10 == 0:
                match_loss, loss = check(net, tdata)
                net.train()
                
            if global_steps % 20 == 0:
                dev_loss, em, f1 = valid()
                agent.append(valid_match_loss, global_steps, dev_loss)
                agent.append(valid_match_em, global_steps, em)
                agent.append(valid_match_f1, global_steps, f1)
                print (global_steps, iepoch, batch, dev_loss, em, f1)
                '''
            '''
                if em > best_em and f1 > best_f1:
                    save_model(net, dev_loss, em, f1, global_steps)
                '''
            '''
                net.train()
           '''   

if __name__ == '__main__':
    
    train()
