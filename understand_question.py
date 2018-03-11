import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import random


class UnderstandQuestion(nn.Module):
    
    def __init__(self, dropout, hidden_size):
        
        super(UnderstandQuestion, self).__init__()
        
        self.hidden_layer = nn.Linear(hidden_size*2, hidden_size)
        self.fc_net = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax()
        
        self.dropout = nn.Dropout(dropout)
        self.cost_func = nn.CrossEntropyLoss()
        
        
    def forward(self, question_encoders, answer_encoders, answer_index):
        
        '''
        question_encoders  batch, qn_steps, hidden_size
        answer_encoders   batch, an_steps, hidden_size
        '''
        
        question_encoders = question_encoders.transpose(0, 1) #qn_steps, batch, hidden_size
        answer_encoders = answer_encoders.transpose(0, 1) #an_steps, batch, hidden_size
        
        print (list(range(answer_index.size(0))))
        print (answer_index.numpy())
        question_encoders = question_encoders[-1]# batch, hidden_size
        answer_encoders = answer_encoders[-1] # batch, hidden_size
        
        print (torch.mean(answer_encoders, 1).cpu().data.numpy())
        answer_encoders = answer_encoders.index_select(0, Variable(answer_index).cuda())
        print (torch.mean(answer_encoders, 1).cpu().data.numpy())
        
        inputs = torch.cat([question_encoders, answer_encoders], -1)
        
        hidden_layer = self.hidden_layer(inputs)
        hidden_layer = self.dropout(hidden_layer)
        
        logits = self.fc_net(hidden_layer)
        _, predictions = torch.max(logits, 1)
        predictions = predictions.cpu().data.numpy()
        
        return logits, predictions
        
        
    def get_loss(self, logits, labels):
        
        labels = Variable(labels).long().cuda()
        loss = self.cost_func(logits, labels)
        
        return loss