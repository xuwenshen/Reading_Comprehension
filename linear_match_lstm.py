import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import random

class MatchLayer(nn.Module):
    
    def __init__(self, hidden_size, dropout):
        
        super(MatchLayer, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.fw_match_lstm = nn.LSTMCell(input_size=hidden_size*2,
                                         hidden_size=hidden_size,
                                         bias=True)
        
        self.bw_match_lstm = nn.LSTMCell(input_size=hidden_size*2,
                                         hidden_size=hidden_size,
                                         bias=True)
        
        self.whq_net = nn.Linear(hidden_size, hidden_size)
        self.whp_net = nn.Linear(hidden_size, hidden_size)
        self.whr_net = nn.Linear(hidden_size, hidden_size)
        self.w_net = nn.Linear(hidden_size, 1)
        
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        
    def forward(self, passage_encoders, question_encoders):
        
        passage_encoders = passage_encoders.transpose(0, 1) # pn_steps, batch, hidden_size
        question_encoders = question_encoders.transpose(0, 1) # qn_steps, batch, hidden_size
        
        
        wq_matrix = self.whq_net(question_encoders) # qn_steps, batch, hidden_size
        wq_matrix = self.dropout(wq_matrix)
        
        wp_matrix = self.whp_net(passage_encoders) # pn_steps, batch, hidden_size
        wp_matrix = self.dropout(wp_matrix)
        
        # forward match lstm (pn_steps, batch, hidden_size)
        fw_match = self.match(passage_encoders, question_encoders, wq_matrix, wp_matrix, fw = True)
        
        
        # backward match lstm (pn_steps, batch, hidden_size)
        bw_match = self.match(passage_encoders, question_encoders, wq_matrix, wp_matrix, fw = False)
        
        match_encoders = torch.cat([fw_match, bw_match], -1) # (pn_steps, batch, hidden_size * 2)
        
        #print ('fw_match.size(): ', fw_match.size())
        #print ('bw_match.size(): ', bw_match.size())
        #print ('match_encoders.size(): ', match_encoders.size())
        
        return match_encoders
        
    def match(self, passage_encoders, question_encoders, wq_matrix, wp_matrix, fw = True):
        
        '''
        passage_encoders (pn_steps, batch, hidden_size)
        question_encoders (qn_steps, batch, hidden_size)
        wq_matrix (qn_steps, batch, hidden_size)
        wp_matrix (pn_steps, batch, hidden_size)
        '''
        if fw:
            match_lstm = self.fw_match_lstm
            start = 0
            end = passage_encoders.size(0)
            stride = 1
        else:
            match_lstm = self.bw_match_lstm
            start = passage_encoders.size(0) - 1
            end = -1
            stride = -1
        
        hx = Variable(torch.zeros(passage_encoders.size(1), self.hidden_size)).cuda()
        cx = Variable(torch.zeros(passage_encoders.size(1), self.hidden_size)).cuda()
        
        match_encoders = [0 for i in range(passage_encoders.size(0))]
        
        for i in range(start, end, stride):
            
            wphp = wp_matrix[i]
            wrhr = self.whr_net(hx)

            _sum = torch.add(wphp, wrhr) # batch, hidden_size
            _sum = _sum.expand(wq_matrix.size(0), wq_matrix.size(1), self.hidden_size) # qn_steps, batch, hidden_size
            
            g = self.tanh(torch.add(wq_matrix, _sum)) # qn_steps, batch, hidden_size

            g = torch.transpose(g, 0, 1)# batch, qn_steps, hidden_size
            
            wg = self.w_net(g) # bactch, qn_steps, 1
            wg = wg.squeeze(-1) # bactch, qn_steps
            alpha = wg # bactch, qn_steps
            alpha = self.softmax(alpha).view(alpha.size(0), 1, alpha.size(1)) # batch,1, qn_steps
            
            
            attentionv = torch.bmm(alpha, question_encoders.transpose(0, 1)) # bacth, 1, hidden_size
            attentionv = attentionv.squeeze(1) # bacth, hidden_size
            
            inp = torch.cat([passage_encoders[i], attentionv], -1)
                        
            hx, cx = match_lstm(inp, (hx, cx)) # batch, hidden_size
            
            match_encoders[i] = hx.view(1, hx.size(0), -1)
            
        match_encoders = torch.cat(match_encoders)
        
        return match_encoders

        
class AnswerLayer(nn.Module):
    
    def __init__(self, hidden_size, dropout, passage_len):
        
        super(AnswerLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.pointer_lstm = nn.LSTMCell(input_size=hidden_size*2,
                                        hidden_size=hidden_size, 
                                        bias=True)
        self.vh_net = nn.Linear(hidden_size*2, hidden_size)
        self.wa_net = nn.Linear(hidden_size, hidden_size)
        self.v_net = nn.Linear(hidden_size, 1)
        
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout()
        self.tanh = nn.Tanh()
        
        self.cost_func = nn.CrossEntropyLoss()
        
    def forward(self, match_encoders):
        
        '''
        match_encoders (pn_steps, batch, hidden_size*2)
        '''
        vh_matrix = self.vh_net(match_encoders) # pn_steps, batch, hidden_size
        
        # prediction start
        h0 = Variable(torch.zeros(match_encoders.size(1), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(match_encoders.size(1), self.hidden_size)).cuda()
        
        wha1 = self.wa_net(h0) # bacth, hidden_size
        wha1 = wha1.expand(match_encoders.size(0), wha1.size(0), wha1.size(1)) # pn_steps, batch, hidden_size
        #print ('_sum.size() ', _sum.size())
        #print ('vh_matrix.size() ', vh_matrix.size())
        f1 = self.tanh(vh_matrix + wha1) # pn_steps, batch, hidden_size
        #print ('f1.size() ', f1.size())
        vf1 = self.v_net(f1.transpose(0, 1)).squeeze(-1) #batch, pn_steps
        
        beta1 = self.softmax(vf1) #batch, pn_steps
        softmax_beta1 = self.softmax(beta1).view(beta1.size(0), 1, beta1.size(1)) #batch, 1, pn_steps
        
        inp = torch.bmm(softmax_beta1, match_encoders.transpose(0, 1)) # bacth, 1, hidden_size
        inp = inp.squeeze(1) # bacth, hidden_size
        
        h1, c1 = self.pointer_lstm(inp, (h0, c0))
        
        
        wha2 = self.wa_net(h1) # bacth, hidden_size
        wha2 = wha2.expand(match_encoders.size(0), wha2.size(0), wha2.size(1)) # pn_steps, batch, hidden_size
        f2 = self.tanh(vh_matrix + wha2) # pn_steps, batch, hidden_size
        vf2 = self.v_net(f2.transpose(0, 1)).squeeze(-1) #batch, pn_steps
        
        beta2 = self.softmax(vf2)#batch, pn_steps
        softmax_beta2 = self.softmax(beta2).view(beta2.size(0), 1, beta2.size(1)) #batch, 1, pn_steps
        
        inp = torch.bmm(softmax_beta2, match_encoders.transpose(0, 1)) # bacth, 1, hidden_size
        inp = inp.squeeze(1) # bacth, hidden_size
        
        h2, c2 = self.pointer_lstm(inp, (h1, c1))
            
        _, start = torch.max(beta1, 1)
        _, end = torch.max(beta2, 1)
        
        beta1 = beta1.view(1, beta1.size(0), beta1.size(1))
        beta2 = beta2.view(1, beta2.size(0), beta2.size(1))
        
        logits = torch.cat([beta1, beta2])
        
        start = start.view(1, start.size(0))
        end = end.view(1, end.size(0))
        
        prediction = torch.cat([start, end]).transpose(0, 1).cpu().data.numpy()
        

        return logits, prediction
    
    
class MatchLSTM(nn.Module):
    
    def __init__(self, hidden_size, dropout, passage_len):
        
        super(MatchLSTM, self).__init__()
        
        self.match = MatchLayer(hidden_size=hidden_size, dropout=dropout)
        self.answer = AnswerLayer(hidden_size=hidden_size, dropout=dropout, passage_len=passage_len)
        
        self.cost_func = None
        
    
    def forward(self, passage_encoders, question_encoders):
        
        match_encoders = self.match(passage_encoders, question_encoders)
        logits, prediction = self.answer(match_encoders)
        
        return logits, prediction
        
    def CrossEntropyLoss(self, logits, labels):
        cost_func = nn.CrossEntropyLoss()
        
        labels = Variable(labels).long().cuda()
        labels = labels.transpose(0, 1)
        loss = (cost_func(logits[0], labels[0])+ cost_func(logits[1], labels[1])) / 2
        
        return loss
   
    def MSELoss(self, logits, labels):
        
        cost_func = nn.MSELoss(size_average =False)
        
        ids = labels.transpose(0, 1)
        ids = ids.contiguous().view(ids.size(0), ids.size(1), 1)
        one_hot = Variable(torch.zeros(logits.size(0), logits.size(1), logits.size(2)).scatter_(-1, ids, 1)).cuda()
        
        loss = cost_func(logits, one_hot)

        return loss


    def get_loss(self, logits, labels):
        
        #return self.CrossEntropyLoss(logits, labels)
        return self.MSELoss(logits, labels)
      
    
        
