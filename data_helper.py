import torch
import nltk
from nltk import tokenize
from nltk.tokenize import TweetTokenizer
import json
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from tqdm import *
from collections import defaultdict
import operator
import random
from nltk.tokenize import WordPunctTokenizer
import h5py

wpt = WordPunctTokenizer()

min_context_len = 20
max_context_len = 350
min_question_len = 2
max_question_len = 30
max_answer_len = 30

def helper(data_path,voc_path, number_data = None):
    
    data = json.load(open(data_path))
    voc = json.load(open(voc_path))

    p_set = []
    p_len_set = []
    p_c_s_e_set = []

    q_set = []
    q_len_set = []
    g_set = []

    a_set = []
    a_len_set = []
    a_b_set = []

    p_str_set = []
    q_str_set = []
    a_str_set = []

    k_set = []

    for i in tqdm(range(len(data))):

        passage = data[i]['passage']
        question = data[i]['question']
        answer_text = data[i]['answertext']
        answer_start = data[i]['answerstart']
        key = data[i]['id']

        answer_end = answer_start + len(answer_text) - 1

        p_tmp = [voc['pad#'] for j in range(max_context_len+1)]
        p_len_tmp = 0
        p_c_s_e_tmp = [[-1, -1] for j in range(max_context_len+1)]

        a_tmp = [voc['pad#'] for j in range(max_answer_len+1)]
        a_len_tmp = 0
        a_b_tmp = [1, 0]

        start = -1
        end = -1

        for wi, (cs, ce) in enumerate(wpt.span_tokenize(passage)):

            if wi == max_context_len: break
                
            if cs <= answer_start and ce >= answer_start:
                start = wi
            if cs <= answer_end and ce >= answer_end:
                end = wi

            p_c_s_e_tmp[wi] = [cs, ce]

            word = passage[cs:ce].lower()
            if word in voc:
                p_tmp[wi] = voc[word]
            else:
                p_tmp[wi] = voc['unk#']
            p_len_tmp = wi + 1

        p_set.append(p_tmp)
        p_len_set.append(p_len_tmp)
        p_c_s_e_set.append(p_c_s_e_tmp)
 
        # 在dev中，如果没有发现完整的answer，就不写入了
        if not (end == -1 or start == -1):
            #在dev中，answer太长，就clip
            end = min(end+1, start+max_answer_len)
            for j in range(start, end+1):
                a_tmp[j-start] = p_tmp[j]
            a_len_tmp = end-start+1
            a_b_tmp = [start, end]

        a_set.append(a_tmp)
        a_len_set.append(a_len_tmp)
        a_b_set.append(a_b_tmp)

        q_tmp = [voc['pad#'] for j in range(max_question_len+2)]
        q_len_tmp = 0
        gtruth_tmp = [voc['pad#'] for j in range(max_question_len + 2)]
        for wi, token in enumerate(wpt.tokenize(question)):
            
            if wi == max_question_len: break
            
            if token.lower() in voc:
                q_tmp[wi] = voc[token.lower()]
                gtruth_tmp[wi+1] = voc[token.lower()]
            else:
                q_tmp[wi] = voc['unk#']
                gtruth_tmp[wi+1] = voc['unk#']
            q_len_tmp = wi + 1

        q_tmp[q_len_tmp] = voc['eos#']
        gtruth_tmp[0] = voc['go#']
        gtruth_tmp[q_len_tmp] = voc['eos#']
        # eos
        q_len_tmp += 1

        q_set.append(q_tmp)
        g_set.append(gtruth_tmp)
        q_len_set.append(q_len_tmp)


        p_str_set.append(passage)
        q_str_set.append(question)
        a_str_set.append(answer_text)
        k_set.append(key)

    
    data = []

    for i in range(len(k_set)):
        tmp_dict = defaultdict(lambda : 0)

        tmp_dict['passage_tokens'] = np.array(p_set[i])
        tmp_dict['passage_len'] = p_len_set[i]
        tmp_dict['char_start_end'] = np.array(p_c_s_e_set[i]) 
        tmp_dict['question_tokens'] = np.array(q_set[i])
        tmp_dict['question_len'] = q_len_set[i]
        tmp_dict['ground_truth'] = np.array(g_set[i])
        tmp_dict['answer_tokens'] = np.array(a_set[i])
        tmp_dict['answer_len'] = a_len_set[i]
        tmp_dict['boundary'] = np.array(a_b_set[i])
        tmp_dict['passage_str'] = p_str_set[i]
        tmp_dict['question_str'] = q_str_set[i]
        tmp_dict['answer_str'] = a_str_set[i]
        tmp_dict['key'] = k_set[i]
        
        tmp_dict = dict(tmp_dict)
        data.append(tmp_dict)

    random.shuffle(data)

    if number_data != None:
        return data[:number_data]
    return data
