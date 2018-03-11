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
from torch.utils import data

from data_helper import helper


class Folder(data.Dataset):
    
    def __init__(self, filepath, voc_path, number_data=None):
        
        self.file = helper(filepath, voc_path, number_data)
        
    def __getitem__(self, index):
        
        return self.file[index]      
        
    def __len__(self):
        return len(self.file)
        
        
