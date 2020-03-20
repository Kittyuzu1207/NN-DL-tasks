# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:11:08 2019

@author: Xinyi Mou
"""


import numpy as np
import copy
import torch
import torch.utils.data as data
import prepro
import pandas as pd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader,Dataset
from deep_model import ESIM

class Batcher(object):
    def __init__(self,embedding_dim=300):
        
        self.train=[]
        self.dev=[]
        self.test=[]
        self.pretrained=True
        self.embedding_dim=embedding_dim
        self.target_dict = {'train': self.train, 'dev': self.dev, 'test': self.test}
        
        
                    
    def prepare(self,dataset):  #using dataset[train]
        word_set=set()
        for i in range(len(dataset['premises'])):
            word_set=set.union(word_set,set(dataset['premises'][i]))
        for i in range(len(dataset['premises'])):
            word_set=set.union(word_set,set(dataset['hypothesis'][i]))
        self.word_dict = dict()
        self.word_dict['_PAD_'] = 0
        self.word_dict['_OOV_'] = 1
        self.V = len(word_set)
        i = 2
        for w in word_set:
            self.word_dict[w] = i
            i += 1
        
    def sen2id(self,sent):
        new_sent=[]
        for w in sent:
            if w in self.word_dict.keys():
                new_sent.append(self.word_dict[w])
            else:
                new_sent.append(self.word_dict['_OOV_'])
        return new_sent
        
    def data2id(self,dataset_name):
        data = self.target_dict[dataset_name]
        for i in range(len(data['premises'])):
            data['premises'][i]=self.sen2id(data['premises'][i])
            data['hypothesis'][i]=self.sen2id(data['hypothesis'][i])
        
    def preprocess(self,dataset):
        self.prepare(dataset['train'])
        for key in self.target_dict.keys():
            self.target_dict[key]=dataset[key]
            self.data2id(key)
        if not self.pretrained:
            return
        w2v_file = open('C:/code/entailment/glove.6B.300d.txt', 'r', encoding='utf-8')
        self.weights = torch.randn([self.V + 2, 300])
        self.weights[0] = 0
        finded = 0
        while True:
            sline = w2v_file.readline()
            if not sline:
                break
            line = sline.strip().split()
            value = [float(v) for v in line[-300:]]
            word = ' '.join(line[:-300])
            if word in self.word_dict.keys():
                ind = self.word_dict[word]
                self.weights[ind] = torch.FloatTensor(value)
                finded += 1
                if finded % 100 == 0:
                    print(str(finded) + '/' + str(self.V) + ' words found')
            else:
                continue
        

        
        
def extend_list(l,target_len):
    return l + [0] * (target_len - len(l))

def collate_fn(data):
    #data.sort(key=lambda x: len(x[0]), reverse=True)
    premises_lens = [len(p['premises']) for p in data]
    hypothesis_lens = [len(p['hypothesis']) for p in data]
    labels =[ p['targets'] for p in data]
    max_plen = max(premises_lens)
    max_hlen = max(hypothesis_lens)
    premises = [extend_list(p['premises'], max_plen) for p in data]
    hypothesis = [extend_list(p['hypothesis'], max_hlen) for p in data]
    return torch.tensor(premises), torch.tensor(hypothesis), torch.tensor(premises_lens), torch.tensor(hypothesis_lens), torch.tensor(labels)
        

class my_dataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        tmp={}
        tmp['premises']=self.data['premises'][index]
        tmp['hypothesis']=self.data['hypothesis'][index]
        tmp['targets']=self.data['targets'][index]
        tmp['pre_len']=self.data['pre_len'][index]
        tmp['hyp_len']=self.data['hyp_len'][index]
        return tmp
#        return self.data[index]

    def __len__(self):
        return len(self.data['premises'])


def eval(model, loss, data_iterator):
    acc = 0
    l = 0
    c = 0
    with torch.no_grad():
        for p, h, p_len, h_len, label in data_iterator:
            prob = model(p, h, p_len, h_len)
            batch_l = loss(prob, label).item()
            batch_a = int(torch.sum(torch.argmax(prob, dim=1) == label))/int(p.shape[0])
            c += 1
            l += batch_l
            acc += batch_a
    return acc/c, l/c

def train():
    print('load_data...')
    model,dataset=prepro.loaddata()
    data_loader=Batcher()
    data_loader.preprocess(dataset)
    train=my_dataset(data_loader.target_dict['train'])
    dev=my_dataset(data_loader.target_dict['dev'])
    test=my_dataset(data_loader.target_dict['test'])
    
    loader = DataLoader(train, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev, batch_size=300, collate_fn=collate_fn)
    test_loader = DataLoader(test, batch_size=300, collate_fn=collate_fn)
    print('init net...')
    net = ESIM(vocab_size=data_loader.V+2, embedding_dim=300, hidden_size=300, num_classes=3, dropout=0.5, pre_trained=data_loader.weights)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    max_epoch = 25
    for epoch in range(max_epoch):
        c=0
        l=0
        acc=0
        net.train()
        for premise,hypothesis,pre_len,hyp_len,targets in loader:
            optimizer.zero_grad()
            logits = net(premise, hypothesis, pre_len, hyp_len)
            batch_loss=loss(logits,targets)
            print(c,batch_loss.item())
            batch_accu = int(torch.sum(torch.argmax(logits, dim=1) == targets))/int(premise.shape[0])
            batch_loss.backward()
            optimizer.step()
            c+=1
            l+=batch_loss.item()
            acc += batch_accu
            if c % 100 == 0:
                print(c, batch_loss, batch_accu)
#            if c==10000:
#                break
        valid_acc, valid_loss = eval(net, loss=loss,data_iterator=dev_loader)
        print('accuracy on dev:', valid_acc, 'loss on dev:', valid_loss)
       
    test_acc, test_loss = eval(net, loss=loss, data_iterator=test_loader)
    print('final accuracy on test:', test_acc, 'loss on test:', test_loss)
    