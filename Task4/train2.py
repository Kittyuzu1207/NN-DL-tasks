# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:53:12 2020

@author: Xinyi Mou
"""

'''
Data Preprocessing
Using CNN+LSTM+CRF to achieve NER task
Referring work in End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF
'''
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader,Dataset
torch.manual_seed(1)
import numpy as np
import codecs 
import pandas as pd
import math
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from models import *
from torch.nn import functional as F


path='E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/8基于LSTM+CRF的序列标注/conll2003/en/ner'
def readfile(path,dataset):
    f=codecs.open(path+'/'+dataset+'.txt')
    data=[]
    label=[]
    tmp1=[]
    tmp2=[]
    for line in f:
        if line != '\n':
            tmp1.append(line.split()[0])
            tmp2.append(line.split()[1])
        else:
            data.append(tmp1)
            label.append(tmp2)
            tmp1=[]
            tmp2=[]
    return data,label


def labelling(train_label,label):
    la=[]
    for i in range(len(train_label)):
        la+=train_label[i]
    la=set(la)
    la_dict={}
    i=0
    for k in la:
        la_dict[k]=i
        i+=1
    for i in range(len(label)):
        for j in range(len(label[i])):
            label[i][j]=la_dict[label[i][j]]
    return label

train_data,train_label=readfile(path,'train')
dev_data,dev_label=readfile(path,'valid')
test_data,test_label=readfile(path,'test')
dev_label=labelling(train_label,dev_label)
test_label=labelling(train_label,test_label)
train_label=labelling(train_label,train_label)
dataset={}
dataset['train']={'word_data':train_data,'char_data':train_data.copy(),'label':train_label}
dataset['dev']={'word_data':dev_data,'char_data':dev_data.copy(),'label':dev_label}
dataset['test']={'word_data':test_data,'char_data':test_data.copy(),'label':test_label}
'''
define batcher
'''
class Batcher(object):
    def __init__(self,embedding_dim=100,char_emb_dim=30):
        self.train=[]
        self.dev=[]
        self.test=[]
        self.pretrained=True
        self.embedding_dim=embedding_dim
        self.char_emb_dim=char_emb_dim
        self.target_dict = {'train': self.train, 'dev': self.dev, 'test': self.test}
        
    def prepare(self,dataset):
        word_set=set()
        char_set=set()
        for i in range(len(dataset['train']['word_data'])):
            word_set=set.union(word_set,set(dataset['train']['word_data'][i]))
            for j in range(len(dataset['train']['word_data'][i])):
                char_set=set.union(char_set,set(list(dataset['train']['word_data'][i][j])))
        self.word_dict = {}
        self.word_dict['_PAD_'] = 0
        self.word_dict['_OOV_'] = 1
        self.V = len(word_set)
        i = 2
        for w in word_set:
            self.word_dict[w] = i
            i += 1
        self.char_size=len(char_set)
        self.char_dict={}
        self.char_dict['_PAD_']=0
        self.char_dict['_OOC_']=1
        i = 2
        for c in char_set:
            self.char_dict[c]=i
            i += 1
            
    def sen2id(self,sent): #sent is a sentence:a list of words
        new_sent=[]
        for w in sent:
            if w in self.word_dict.keys():
                new_sent.append(self.word_dict[w])
            else:
                new_sent.append(self.word_dict['_OOV_'])
        return new_sent
    
    def char2id(self,sent):
        new_sent=[]
        for i in range(len(sent)):
            tmp=[]
            for c in list(sent[i]):
                if c in self.char_dict.keys():
                    tmp.append(self.char_dict[c])
                else:
                    tmp.append(self.char_dict['_OOC_'])
            new_sent.append(tmp)
        return new_sent
        
    
    def data2id(self,dataset_name):
        data = self.target_dict[dataset_name]
        for i in range(len(data['word_data'])):
            data['word_data'][i]=self.sen2id(data['word_data'][i])
        for i in range(len(data['char_data'])):
            data['char_data'][i]=self.char2id(data['char_data'][i])
    
    def preprocess(self,dataset):
        self.prepare(dataset)
        for key in self.target_dict.keys():
            self.target_dict[key]=dataset[key]
            self.data2id(key)
        if not self.pretrained:
            return
        w2v_file = open('E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/8基于LSTM+CRF的序列标注/glove100d.txt', 'r', encoding='utf-8')
        self.weights = torch.randn([self.V + 2, 100])
        self.weights[0] = 0
        finded = 0
        while True:
            sline = w2v_file.readline()
            if not sline:
                break
            line = sline.strip().split()
            value = [float(v) for v in line[-100:]]
            word = ' '.join(line[:-100])
            if word in self.word_dict.keys():
                ind = self.word_dict[word]
                self.weights[ind] = torch.FloatTensor(value)
                finded += 1
                if finded % 100 == 0:
                    print(str(finded) + '/' + str(self.V) + ' words found')
            else:
                continue

def extend_list(l,target_len):
    if len(l)<=target_len:
        return l + [0] * (target_len - len(l))
    else:
        return l[:target_len]

def extend_label(l,target_len):
    return l+ [9] *(target_len-len(l))

def extend_word(l,max_slen,max_wlen):
    return l+[[0]*max_wlen]*(max_slen-len(l))


def collate_fn(data):
    sent_lens = [len(p['word_data']) for p in data]  #[5,6]
    word_lens = [[len(w) for w in p['char_data']] for p in data] #[[2,3,2,4,5],[2,3,4,4,3,2]]
    labels =[p['label'] for p in data]
    max_slen = max(sent_lens)
#    tmp=[]
#    for w in word_lens:
#        tmp+=w
#    max_wlen = max(tmp)
    max_wlen=20
    sent = [extend_list(p['word_data'], max_slen) for p in data]  
    words =[[extend_list(w,max_wlen) for w in p['char_data']] for p in data]
    words=[extend_word(p,max_slen,max_wlen) for p in words]
    labels=[extend_label(p['label'],max_slen) for p in data]
    word_lens=[extend_list(w,max_slen) for w in word_lens]
    mask=torch.tensor(sent)>0
    return torch.tensor(sent), torch.tensor(words),torch.tensor(mask), torch.tensor(sent_lens), torch.tensor(labels)
           
    

class my_dataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        tmp={}
        tmp['word_data']=self.data['word_data'][index]
        tmp['char_data']=self.data['char_data'][index]
        tmp['label']=self.data['label'][index]
        
        return tmp
#        return self.data[index]
    def __len__(self):
        return len(self.data['word_data'])



def train():
    print('load_data...')
    data_loader=Batcher()
    data_loader.preprocess(dataset)
    train=my_dataset(data_loader.target_dict['train'])
    dev=my_dataset(data_loader.target_dict['dev'])
    test=my_dataset(data_loader.target_dict['test'])
    
    loader = DataLoader(train, batch_size=10, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev, batch_size=300, collate_fn=collate_fn)
    test_loader = DataLoader(test, batch_size=300, collate_fn=collate_fn)
    print('init net...')
    args={'vocab_size':data_loader.V+2,
          'char_size':data_loader.char_size+2,
          'embed_dim':100,
          'char_emb_dim':30,
          'label_num':11,
          'paddingId':0,
          'char_paddingId':0,
          'pretrained_embed':True,
          'pretrained_weight':data_loader.weights,
          'device':'cpu',
          'conv_filter_sizes':[3],
          'conv_filter_nums':[30],
          'hidden_size':200,  
          'num_layers':1,
          'dropout':0.5 
            }
    net=CNN_BiLSTM(args)
    args={
            'target_size':9,
            'device':'cpu'
            }
    crf=CRF(args)
    loss_func= crf.neg_log_likelihood
    optimizer = torch.optim.Adam([{'params':net.parameters(),'params':crf.parameters()}],lr=0.015)
    max_epoch=20
    
    for epoch in range(max_epoch):
        net.train()
        c=0
#        gold_labels = []
#        predict_labels = []
        for sen,word,mask,sen_len,label in loader:
            gold_labels = []
            predict_labels = []
            optimizer.zero_grad()
            logits=net(sen,word,sen_len) 
            logits_tmp=torch.empty(logits.size())
            for i in range(len(logits)):
                logits_tmp[i]=F.softmax(logits[i],dim=1)
            path_score, best_paths = crf(logits_tmp, mask)
            batch_loss=loss_func(logits_tmp,mask,label)
            print(c,batch_loss.item())
            batch_loss.backward()
            optimizer.step()
            c+=1
            for i in range(len(sen_len)):
                gold_labels.append(label[i][:sen_len[i]])
                predict_labels.append(best_paths[i][:sen_len[i]])
            if c %2000==0:
                print('acc:',get_acc(predict_labels,gold_labels))
                print('P R F:')
                print(eval_data(predict_labels,gold_labels))
        
        


    
def eval_data(predict_labels,gold_labels):
    TP={'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0}
    TN={'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0}
    FP={'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0}
    FN={'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0}
    precision={}
    recall={}
    for i in range(len(gold_labels)):
        for j in range(len(gold_labels[i])):
            for key in TP.keys():
                if gold_labels[i][j]==int(key) and predict_labels[i][j]==int(key):
                    TP[key]+=1
                elif gold_labels[i][j]==int(key) and predict_labels[i][j]!=int(key):
                    FN[key]+=1
                elif gold_labels[i][j]!=int(key) and predict_labels[i][j]!=int(key):
                    TN[key]+=1
                else:
                    FP[key]+=1
    for key in TP.keys():
        try:
            precision[key]=TP[key]/(TP[key]+FP[key])
        except:
            precision[key]=0
        try:
            recall[key]=TP[key]/(TP[key]+FN[key])
        except:
            recall[key]=0
    total_P=sum(precision.values())/len(TP.keys())
    total_R=sum(recall.values())/len(TP.keys())
    F1=2*total_P*total_R/(total_P+total_R)
    
    return total_P,total_R,F1
        
                
def get_acc(predict_labels,gold_labels):
    acc=0
    num=0
    for i in range(len(gold_labels)):
        acc+=sum(gold_labels[i]==predict_labels[i])
        num+=len(gold_labels[i])
    acc=acc.item()/num

    return acc      
                
def eval(net,crf,loss,data_iter):
    acc = 0
    l = 0
    c = 0
    p=0
    r=0
    f=0
    with torch.no_grad():
        for sen,word,mask,sen_len,label in data_iter:
            gold_labels = []
            predict_labels = []
            logits=net(sen,word,sen_len)            
            path_score, best_paths = crf(logits, mask) 
            for i in range(len(sen_len)):
                gold_labels.append(label[i][:sen_len[i]])
                predict_labels.append(best_paths[i][:sen_len[i]])
            batch_l = loss(logits,mask, label).item()
            batch_a = get_acc(predict_labels,gold_labels)          
            p1,r1,f1= eval_data(predict_labels,gold_labels)
            p+=p1
            r+=r1
            f+=f1
            c += 1
            l += batch_l
            acc += batch_a
    return acc/c, l/c,p/c,r/c,f/c
                    
def main():
    train()

if __name__=='__main__':
    main()        
    