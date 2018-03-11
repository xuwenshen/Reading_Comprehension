import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import random

from understand_passage import UnderstandPassage
from understand_question import UnderstandQuestion
from linear_match_lstm import MatchLSTM
from preprocess import PreprocessLayer


min_context_len = 20
max_context_len = 350
min_question_len = 2
max_question_len = 30
max_answer_len = 30

passage_rindex = torch.Tensor(range(max_context_len-1, -1, -1)).long()
question_rindex = torch.Tensor(range(max_question_len-1, -1, -1)).long()
answer_rindex = torch.Tensor(range(max_answer_len-1, -1, -1)).long()


class MultiTask(nn.Module):
    
    def __init__(self, hidden_size, dropout, passage_len, embedding_path, voc_path):
        
        super(MultiTask, self).__init__()

        self.hidden_size = hidden_size
        
        self.embedding = None
        self.embedding_dim = None
        self.init_embedding(embedding_path)
        self.voc = json.load(open(voc_path))
        
        self.preprocess_layer = PreprocessLayer(hidden_size, dropout, self.embedding_dim)
        self.understandpassage_task = UnderstandPassage(self.embedding, hidden_size, dropout, self.voc)
        self.understandquestion_task = UnderstandQuestion(dropout, hidden_size)
        self.match_task = MatchLSTM(hidden_size, dropout, passage_len)
        
        self.dropout = nn.Dropout(dropout)
        
        
    def init_embedding(self, embedding_path):
        
        pretrained_weight = torch.Tensor(json.load(open(embedding_path)))
        embedding = torch.nn.Embedding(num_embeddings=pretrained_weight.size(0), embedding_dim=pretrained_weight.size(1))
        embedding.weight = nn.Parameter(pretrained_weight)
        embedding.weight.requires_grad = False

        self.embedding_dim = pretrained_weight.size(1)
        self.embedding = embedding
        
        
    def preprocess(self, passage, question, answer = None):
        
        #passage 不能reverse，因为label对应关系
        # reverse
        question = question.index_select(1, question_rindex)
        
        passage = Variable(passage).long().cuda()
        question = Variable(question).long().cuda()
        
        passage = self.embedding(passage)
        question = self.embedding(question)
        
        if not answer is None:
            answer = answer.index_select(1, answer_rindex)
            answer = Variable(answer).long().cuda()
            answer = self.embedding(answer)
        
        passage_encoders, question_encoders, answer_encoders = self.preprocess_layer(passage, question, answer)
        return passage_encoders, question_encoders, answer_encoders
    

    def forward(self, passage, question, answer_index = None, answer = None, decoder_inputs = None, max_question_len = 0, is_generation = False, is_classification = False, is_teacher_forcing = True):
        
        passage_encoders, question_encoders,  answer_encoders = self.preprocess(passage, question, answer)
        
        generation_logits = None
        generation_predictions = None
        classification_logits = None
        classification_predictions = None
        match_logits = None
        match_predictions = None
        
        if is_generation:
            generation_logits, generation_predictions = self.understandpassage_task(passage_encoders, answer_encoders, decoder_inputs, is_teacher_forcing, max_question_len)
            
        if is_classification:
            classification_logits, classification_predictions = self.understandquestion_task(question_encoders, answer_encoders, answer_index)
            
        match_logits, match_predictions = self.match_task(passage_encoders, question_encoders)
        
        return {'generation_logits':generation_logits,
                'generation_predictions':generation_predictions,
                'classification_logits':classification_logits,
                'classification_predictions':classification_predictions,
                'match_logits':match_logits,
                'match_predictions':match_predictions}
    
    def get_loss(self, match_logits, match_labels, generation_logits=None, generation_labels=None, classification_logits=None, classification_labels=None, is_generation = False, is_match = True, is_classification = False, lambda_m=1, lambda_g=1, lambda_c = 1):
        
        generation_loss = 0
        match_loss = 0
        classification_loss = 0
        loss = 0
        return_ = {'generation_loss':0, 'match_loss':0,'classification_loss':0,'loss':0}
        
        if is_match:
            match_loss = self.match_task.get_loss(match_logits, match_labels)
            return_['match_loss'] = sum(match_loss.cpu().data.numpy())
            loss += lambda_m * match_loss
        
        if is_generation:
            generation_loss = self.understandpassage_task.get_loss(generation_logits, generation_labels)
            loss += lambda_g * generation_loss
            return_['generation_loss'] = sum(generation_loss.cpu().data.numpy())
            
        if is_classification:
            classification_loss = self.understandquestion_task.get_loss(classification_logits, classification_labels)
            loss += lambda_c * classification_loss
            return_['classification_loss'] = sum(classification_loss.cpu().data.numpy())
            
        return_['loss'] = loss
        
        return return_