import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import random

class UnderstandPassage(nn.Module):
    
    def __init__(self, embedding, hidden_size, dropout, voc):
        
        
        super(UnderstandPassage, self).__init__()
        
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.embedding_dim = self.embedding.weight.size(1)
        self.voc = voc
        self.voc_size = len(voc)
        
        self.dropout = nn.Dropout(dropout)
        
        #encode passage_encoders, answer_encoders concatenation
        self.enc_net = nn.LSTM(input_size=hidden_size*2,
                                hidden_size=hidden_size, 
                               bidirectional = True,
                                bias=True)
        
        self.dec_net = nn.LSTM(input_size=self.embedding_dim+hidden_size*2,
                                hidden_size=hidden_size, 
                                bias=True)
        
        self.fc_net = nn.Linear(hidden_size, self.voc_size)
    
        self.cost_func = self.init_cost_func()
        
    def init_cost_func(self):
        
        weight = [1 for i in range(len(self.voc))]
        weight[self.voc['pad#']] = 0
        
        return nn.CrossEntropyLoss(weight=torch.Tensor(weight))
        
    def enc(self, passage_encoders, answer_encoders):
        
        '''
        answer_encoders an_steps, batch, hidden_size
        passage_encoders pn_steps, batch, hidden_size
        '''
        answer_encoders = answer_encoders[-1] # batch, hidden_size
        answer_encoders = answer_encoders.expand(passage_encoders.size(0), passage_encoders.size(1), self.hidden_size)
        inputs = torch.cat([passage_encoders, answer_encoders], -1) # pn_steps, batch, hidden_size*2
        
        encoders, hidden = self.enc_net(inputs) # pn_steps, batch, hidden_size*2
        encoders = self.dropout(torch.mean(encoders, 0).squeeze(0)) #  batch, hidden_size*2
        
        return encoders
        
    def dec(self, encoders, decoder_inputs, is_teacher_forcing, max_question_len):
        
        '''
        encoders (batch, hidden_size)
        if is_teacher_forcing: decoder_inputs (batch, max_question_len)
        if not is_teacher_forcing: decoder_inputs (batch, 1)
        '''
        decoder_inputs = Variable(decoder_inputs).long().cuda()
        decoder_inputs = self.embedding(decoder_inputs)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        
        encoders = encoders.expand(decoder_inputs.size(0), encoders.size(0), self.hidden_size*2)
        inputs = torch.cat([decoder_inputs, encoders], -1)
        
        if is_teacher_forcing:
            
            outputs, hidden = self.dec_net(inputs)
            outputs = self.dropout(outputs)
            logits = self.fc_net(outputs) # qn_steps, batch, voc_size
            
            _, predictions = torch.max(logits.transpose(0, 1), -1) #batch, qn_steps
            predictions = predictions.cpu().data.numpy()
            
        else:
            logits = [0 for i in range(max_question_len)]
            predictions = [0 for i in range(max_question_len)]
            
            output, hidden = self.dec_net(inputs)
            output = self.dropout(output)
            logits[0] = self.fc_net(output)
            
            _, index = torch.max(logits[0])
            
            logits[0] = logits[0].view(1, decoder_inputs.size(1), self.voc_size) # 1，batch_size, voc_size
            predictions[0] = index.cpu().data.numpy() # batch_size
            
            for i in range(1, max_question_len):
                
                prev_output = Variable(predictions[i-1]).long().cuda()
                prev_output = self.embedding(prev_output)
                inputs = torch.cat([prev_output, encoders[0]], -1)
                
                output, hidden = self.dec_net(inputs, hidden)
                output = self.dropout(output)
                logits[i] = self.fc_net(output)

                _, index = torch.max(logits[i])
                
                logits[i] = logits[i].view(1, decoder_inputs.size(0), self.voc_size) # 1，batch_size, voc_size
                predictions[i] = index.cpu().data.numpy() # batch_size
            
            logits = torch.cat(logits)# qn_steps, batch, voc_size
            predictions = np.array(predictions).transpose(1, 0)
            
        return logits, predictions
            
        
    def forward(self, passage_encoders, answer_encoders, decoder_inputs=None, is_teacher_forcing = True, max_question_len=0):
        
        '''
        answer_encoders (batch, an_steps, hidden_size)
        passage_encoders (batch, pn_steps, hidden_size)
        
        if is_teacher_forcing: decoder_inputs (batch, max_question_len)
        if not is_teacher_forcing: decoder_inputs (None)
        '''
        
        passage_encoders = passage_encoders.transpose(0, 1) # pn_steps, batch, hidden_size
        answer_encoders = answer_encoders.transpose(0, 1) # an_steps, batch, hidden_size
        
        encoders = self.enc(passage_encoders, answer_encoders[-1])# batch, hidden_size
        
        if decoder_inputs is None:
            decoder_inputs = torch.Tensor([[self.voc['go#']]]*encoders.size(0))
        logits, predictions = self.dec(encoders, decoder_inputs, is_teacher_forcing, max_question_len)
        
        
        return logits, predictions
        
    def get_loss(self, logits, labels):
        
        labels = Variable(labels).long().cuda()
        labels = labels.transpose(0, 1)
        
        logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)
        
        loss = torch.mean(self.cost_func(logits, labels))
        
        return loss
    
        
        
        