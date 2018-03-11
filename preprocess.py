import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import random

class PreprocessLayer(nn.Module):
    
    def __init__(self, hidden_size, dropout, embedding_dim):
        
        super(PreprocessLayer, self).__init__()
        
        self.passage_lstm = nn.LSTM(input_size=embedding_dim,
                                    hidden_size=hidden_size,
                                    num_layers=1,
                                    dropout=dropout,
                                    batch_first=True)
        
        self.question_lstm = nn.LSTM(input_size=embedding_dim,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     dropout=dropout,
                                     batch_first=True)
        
        self.answer_lstm = nn.LSTM(input_size=embedding_dim,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     dropout=dropout,
                                     batch_first=True)
        
        
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, passage, question, answer = None):
        
        passage_encoders = None
        question_encoders = None
        answer_encoders = None
        
        passage_encoders, p_states = self.passage_lstm(passage)
        question_encoders, q_states = self.question_lstm(question)
        
        if not answer is None:
            answer_encoders, a_states = self.answer_lstm(answer)
        
        return passage_encoders, question_encoders, answer_encoders
