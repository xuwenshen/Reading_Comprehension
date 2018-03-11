import numpy as np 
from tqdm import *
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import random
import h5py
import sys, os
from collections import defaultdict
import nltk
import string


def revers_dic(dic):
    
    rdic = defaultdict()
    for (k,v) in dic.items():
        rdic[v] = k
    rdic = dict(rdic)
    return rdic
        
    
class Transform:
    
    def __init__(self, voc_path):
        
        self.voc = json.load(open(voc_path))
        self.rvoc = revers_dic(self.voc)
        
        self.go_id = self.voc['go#']
        self.eos_id = self.voc['eos#']
        self.pad_id = self.voc['pad#']
        self.unk_id = self.voc['unk#']
        
        
        self.go = 'go#'
        self.eos = 'eos#'
        self.pad = 'pad#'
        self.unk = 'unk#'
        
    def i2t(self, lst):
        text = ''
        for i in range(len(lst)):
            token = self.rvoc[lst[i]]
            if token == self.eos:
                break
            if token in string.punctuation:
                text += self.rvoc[lst[i]]
            else:
                text += ' ' + self.rvoc[lst[i]]

        text = text.lstrip(' ')  
        return text