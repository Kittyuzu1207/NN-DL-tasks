# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:04:09 2019

@author: Xinyi Mou
"""

#Word Embedding  
import numpy as np
import pandas as pd
import random
#train=pd.read_csv('train.tsv', sep='\t')
test=pd.read_csv('test.tsv', sep='\t')
##划分训练集和验证集
#dev=train.sample(frac=0.3,random_state=0,axis=0)
#train=train[~train.index.isin(dev.index)]
#train.to_csv('train.csv')
#dev.to_csv('val.csv')
train=pd.read_csv('train.csv')
dev=pd.read_csv('val.csv')


x_train=list(train['Phrase'])
y_train=list(train['Sentiment'])
x_dev=list(dev['Phrase'])
y_dev=list(dev['Sentiment'])
x_test=list(test['Phrase'])
#全部转成小写
def lower(data):
    for i in range(len(data)):
        data[i]=data[i].lower()
    return data

x_train=lower(x_train)
x_dev=lower(x_dev)
x_train=[x.split() for x in x_train]
x_dev=[x.split() for x in x_dev]


#f=open('E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/6基于深度学习的文本分类/x_train.txt','a')
#for line in x_train:
#    f.write(line+'\n')
#f.close()  #写入文件后给glove训练

#取语料中top 5000 words 作为nn训练时的词典范围，其他的用0替代
#def stopwordslist(filepath):  
#    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]  
#    return stopwords 


#method1:word embedding using w2v
from gensim.models import Word2Vec 
   
def pretrain_w2v(x_train):
    model1 = Word2Vec(x_train,sg=1, size=96, window=5, min_count=1, workers=4)
    pretrained_vec=model1.wv[model1.wv.index2word[0]]
    for i in range(1,len(model1.wv.index2word)):
        pretrained_vec=np.vstack((pretrained_vec,model1.wv[model1.wv.index2word[i]]))
    return pretrained_vec,model1
    
    
#method2:random embedding

#随机初始化lookup table
#直接在训练torch.NN时使用nn.embedding


#method2:word embedding using glove
#pre-train using glove code provided by Stanford
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
def pretrain_glove():
    glove_file = datapath('E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/6基于深度学习的文本分类/vectors.txt')
    tmp_file = get_tmpfile("test_word2vec.txt")
    glove2word2vec(glove_file, tmp_file)
    model3 = KeyedVectors.load_word2vec_format(tmp_file)
    pretrained_vec=model3.wv[model3.wv.index2word[0]]
    for i in range(1,len(model3.wv.index2word)):
        pretrained_vec=np.vstack((pretrained_vec,model3.wv[model3.wv.index2word[i]]))
    return pretrained_vec,model3


#method3:fasttext
from gensim.models import FastText
def pretrain_fasttext(x_train):
    model = FastText(x_train,min_count=1,size=32)
    pretrained_vec=model.wv[model.wv.index2word[0]]
    for i in range(1,len(model.wv.index2word)):
        pretrained_vec=np.vstack((pretrained_vec,model.wv[model.wv.index2word[i]]))
    return pretrained_vec,model
    