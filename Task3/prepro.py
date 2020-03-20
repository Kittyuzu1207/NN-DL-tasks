# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:50:51 2019

@author: Xinyi Mou
"""

#preprocessing
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import codecs
import numpy as np
import pandas as pd
import torch


def readfile(path):
    f=codecs.open(path)
    temp=[]
    for line in f:
        temp.append(line.split('\t'))
    f.close()
    label=[temp[i][0] for i in range(1,len(temp))]
    out=[]
    for i in range(len(temp)):
        if temp[i][0] =='-':
            out.append(i)
    ls=list(set(list(range(1,len(temp))))-set(out))
    label=[temp[i][0] for i in ls]
    label_idx={'contradiction':2,'neutral':0,'entailment':1}
    label=[label_idx[i] for i in label]
    sentence1=[temp[i][5] for i in ls]
    #对hypothesis句前添加一个NULL关键字
    sentence2=[temp[i][6] for i in ls]
    return label,sentence1,sentence2




def sentence2words(sentence):
    sentence=sentence.lower()
    punctuations = [".", ",", ";", "!", "?", "/", '"', "'", "(", ")", "{", "}", "[", "]", "="]
    for punctuation in punctuations:
        sentence = sentence.replace(punctuation, " {} ".format(punctuation))
    sentence = sentence.replace("  ", " ")
    sentence = sentence.replace("   ", " ")
    sentence = sentence.split(" ")
    todelete = ["", " ", "  "]
    for i, elt in enumerate(sentence):
        if elt in todelete:
            sentence.pop(i)
    return sentence

def loaddata(path='C:/code/entailment/'
,glove_path='C:/code/entailment/glove.6B.300d.txt'):
    print("loading glove vector...")
    glove_file = datapath('C:/code/entailment/glove.6B.300d.txt')
    tmp_file = get_tmpfile("test_word2vec.txt")
    glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)
    print("glove done")
#    print("loading word2vec...")
#    model =KeyedVectors.load_word2vec_format('D:/BaiduNetdiskDownload/word2vec/GoogleNews-vectors-negative300.bin',binary=True)
#    print('word2vec done')
       
    dataset={}
    print("loading dataset")
    for type_set in ['train','dev','test']:
        y,pre,hyp=readfile(path+'snli_'+type_set+'.txt')
#        y=y[:5000]
#        pre=pre[:5000]
#        hyp=hyp[:5000]
        dataset[type_set]={"premises":pre,"hypothesis":hyp,"targets":y}   
    tokenized_dataset = simple_preprocess(dataset, model)
    print('dataset done')
#    pre=tokenized_dataset['train']
#    words=[]
#    for w in pre['premises']:
#        words+=w
#    for w in pre['hypothesis']:
#        words+=w
#    words=list(set(words))
#    yes=[]
#    dic={}
#    for w in words:
#        if w in model.index2word:
#            yes.append(w)
#    for w in yes:
#        dic[w]=model[w]   
#    dic['_']=model['_']
    return model,tokenized_dataset

def simple_preprocess(dataset,model):
    tokenized_dataset = dict((type_set, {"premises": [], "hypothesis": [], "targets": [],'pre_len':[],'hyp_len':[]}) for type_set in dataset)
    print("tokenization")
    for type_set in dataset:
        print("type_set:"+type_set)
        num_ids = len(dataset[type_set]["targets"])
        for i in range(num_ids):
            try:
                premises_tokens=sentence2words(dataset[type_set]['premises'][i])
                hypothesis_tokens=sentence2words(dataset[type_set]['hypothesis'][i])
                target=dataset[type_set]['targets'][i]

            except:
                pass
            else:
                tokenized_dataset[type_set]["premises"].append(premises_tokens)
                tokenized_dataset[type_set]["hypothesis"].append(hypothesis_tokens)
                tokenized_dataset[type_set]["targets"].append(target)
                tokenized_dataset[type_set]["pre_len"].append(len(premises_tokens))
                tokenized_dataset[type_set]["hyp_len"].append(len(hypothesis_tokens))
    print("tokenization done")
    return tokenized_dataset
    
    
    
    
    
