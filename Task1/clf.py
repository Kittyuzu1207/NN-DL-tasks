# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 14:59:51 2019

@author: Xinyi Mou
"""


'''
This file is aimed at text classification based on machine learning
Several parts are included:
part1 preprocessing
      Read the dataset and extract features
part2 classifiers contruction
      logistic/softmax regression
      define loss function
      training through gradient descent
      Feature selection
'''

import numpy as np
import pandas as pd
import random
train=pd.read_csv('train.tsv', sep='\t')
test=pd.read_csv('test.tsv', sep='\t')
#划分训练集和验证集
dev=train.sample(frac=0.3,random_state=0,axis=0)
train=train[~train.index.isin(dev.index)]

sentences=list(train['Phrase'])+list(dev['Phrase'])

def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]  
    return stopwords 
stoplist=stopwordslist('E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/5基于机器学习的文本分类/stopword.txt')


#Feature Extraction
#1. Bag-of-word   1,2,3-gram
#CountVectorizer of sklearn
from sklearn.feature_extraction.text import CountVectorizer
co= CountVectorizer(
    analyzer='word',
    lowercase=True,
    ngram_range=(1,3),    #考虑1，2，3gram
    stop_words='english',   #使用内置停用词
    max_features=500
)
co.fit(sentences)
#train_bow = co.transform(list(train['Phrase']))
#dev_bow=co.transform(list(dev['Phrase']))
#test_bow = co.transform(list(test['Phrase']))

#考虑用tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=tf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1,3),
    max_features=500
)
tfidf.fit(sentences)
#train_tfidf=tfidf.transform(list(train['Phrase']))
#dev_tfidf=tfidf.transform(list(dev['Phrase']))
#test_tfidf=tfidf.transform(list(test['Phrase']))

#2.高频词频率
word_list=[]
for sen in sentences:
   tmp=[]
   for w in sen.split(' '):
       if w.lower() not in stoplist and w not in stoplist:
           tmp.append(w)
   word_list+=tmp

import nltk
all_words = nltk.FreqDist(w for w in word_list)
most_common_word = all_words.most_common(500)
high_fre_words=[t[0] for t in most_common_word]
high_fre_dict=dict(zip(high_fre_words,range(0,len(high_fre_words))))


def get_high_fre(sen):   #获得一个句子的这些高频词词频
    fre=np.zeros((1,len(high_fre_words)))
    tmp=sen.split(' ')
    tmp=[w.lower() for w in tmp]
    for w in tmp:
        if w in high_fre_words:
            fre[0,high_fre_dict[w]]+=1
    return fre

def high_fre(sen):   #高频词是否出现
    fre=np.zeros((1,len(high_fre_words)))
    tmp=sen.split(' ')
    tmp=[w.lower() for w in tmp]
    for w in tmp:
        if w in high_fre_words:
            fre[0,high_fre_dict[w]]=1
    return fre


#3.The number of sentiment words
#sentiment dictionary from HowNet
pos_senti=stopwordslist('E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/5基于机器学习的文本分类/pos_senti.txt')
neg_senti=stopwordslist('E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/5基于机器学习的文本分类/neg_senti.txt')
#情感词是否出现
def senti_word_fre(sen,pos,neg):
    tmp=sen.split(' ')   
    fre=np.zeros((1,2))
    p=0
    n=0
    for w in tmp:
        if w in pos:
            p=1
            break
    for w in tmp:
        if w in neg:
            n=1
            break
    fre[0,0]=p
    fre[0,1]=n
    return fre

#4.The appearance of positive and negative words
postive=stopwordslist('E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/5基于机器学习的文本分类/postive.txt')
negative=stopwordslist('E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/5基于机器学习的文本分类/negative.txt')


#4.The existence of Not words and degree words
not_word=stopwordslist('E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/5基于机器学习的文本分类/not.txt')
degree=stopwordslist('E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/5基于机器学习的文本分类/degree.txt')
def not_degree_appear(sen):
    tmp=sen.split(' ')
    fe=np.zeros((1,2))
    for w in tmp:
        if w in not_word:
            fe[0,0]=1
            break
    for w in tmp:
        if w in degree:
            fe[0,1]=1
            break
    return fe


#合并特征
def get_feature(sen):
    f_tfidf=tfidf.transform([sen])
    f_high_fre=get_high_fre(sen)
    f_senti_words=senti_word_fre(sen,pos_senti,neg_senti)
    f_p_n_words=senti_word_fre(sen,postive,negative)
    f_n_d=not_degree_appear(sen)
    feature=np.hstack((f_tfidf.toarray(), f_high_fre))
    feature=np.hstack((feature,f_senti_words))
    feature=np.hstack((feature,f_p_n_words))
    feature=np.hstack((feature,f_n_d))
    return feature
    
def load_data(x_data):  #把句子转化为特征   x_data 是list，每一个元素是一个句子
    feature=get_feature(x_data[0])
    for i in range(1,len(x_data)):
        feature=np.vstack((feature,get_feature(x_data[i])))
    return feature

#Classifier construction
#Logistic/Softmax Regression  多分类用softmax回归
#p(y=c|x)=softmax(wcTx)    y/hat=argmax(wcTx)
    
class SoftmaxRegression:
    def __init__(self):
        self.weight= None  # 权值
        self.C=None #类别个数
        self.D=None #输入数据维度
        self.N = None  # 样本总量
        
    def softmax(self,X):
        return np.exp(X)/np.sum(np.exp(X),axis=1,keepdims=True)
    
    def init_param(self,x_train,y_train):
        b = np.ones((x_train.shape[0], 1))
        x_train = np.hstack((x_train, b))  # 附加偏置项
        self.C=len(np.unique(y_train))
        self.D = x_train.shape[1]
        self.N = x_train.shape[0]
        self.weight=np.ones((self.C, self.D))
        return x_train
    
    def loss_func(self,y_label,y_prob,weight,lamda=0.01):
        weight_sq=[[weight[i][j]**2 for j in range(len(weight[i]))] for i in range(len(weight))]
        loss=-(1/self.N)*np.sum(y_label*np.log(y_prob))+lamda/2*np.sum(weight_sq)
        #加上正则项  限制权重  避免计算softmax时溢出
        return loss
    

    def one_hot(self,y_train):
        one_hot=np.zeros((self.N,self.C))
        one_hot[np.arange(self.N),np.array(y_train).T]=1
        return one_hot

    def bgd(self,x_train,y_train, maxIter=1000, alpha = 0.1,lamda=0.01):  #bgd  批量梯度下降
        x_train=self.init_param(x_train,y_train)
#        weight=np.zeros((self.weight.shape[0],self.weight.shape[1]))
        weight=np.random.rand(self.weight.shape[0],self.weight.shape[1])
        Iter=0
        loss_list=[]
        while Iter<maxIter:
            print('Iter:'+str(Iter))
            gradient=0
            y=self.one_hot(y_train)
            y_prob=self.softmax(np.dot(x_train,weight.T))
            gradient = -1/(self.N)*np.dot((y-y_prob).T,x_train)+lamda*weight
            loss=self.loss_func(y,y_prob,weight,lamda)
            weight-=alpha*gradient
            print('loss:'+str(loss))
            loss_list.append(loss)
            Iter+=1
        self.weight=weight
        return self.weight,loss_list

#梯度下降的变形：批量batch，随机stochastic，小批量mini-bacth     
    def sgd(self,x_train,y_train,maxIter=1000,alpha=0.1,lamda=0.01):   #随机梯度下降  每次只用一个样本
        x_train=self.init_param(x_train,y_train)
        weight=np.random.rand(self.weight.shape[0],self.weight.shape[1])
        Iter=0
        while Iter<maxIter:
            print('Iter:'+str(Iter))
            y_one_hot=self.one_hot(y_train)
            y_prob=self.softmax(np.dot(x_train,weight.T))
            loss=self.loss_func(y_one_hot,y_prob,weight,lamda)  #更新只用一个样本，但是算loss还是所有样本
#            x=x_train[j].reshape(-1,1)
            j=random.randint(0,self.N)
            x=x_train[j].reshape(len(x_train[j]),1)
            y=self.one_hot(y_train)[j]
            gradient=-np.dot((y-y_prob[j]).reshape(len(y),1),x.T)+lamda*weight
            weight-=alpha*gradient
            print('loss:'+str(loss))
            Iter+=1
        self.weight=weight
        return self.weight
  
    
    def mgd(self,x_train,y_train,batch_size=10,maxIter=1000,alpha=0.1,lamda=0.01):   #小批量梯度下降 每次只用样本的一个子集训练
        x_train=self.init_param(x_train,y_train)
        weight=np.random.rand(self.weight.shape[0],self.weight.shape[1])
        Iter=0
        i=0
        while Iter<maxIter:
           print('Iter:'+str(Iter))
           x=x_train[i*batch_size:(i+1)*batch_size]
           y_one_hot=self.one_hot(y_train)
           y=y_one_hot[i*batch_size:(i+1)*batch_size]
           y_prob=self.softmax(np.dot(x_train,weight.T))
           loss=self.loss_func(y_one_hot,y_prob,weight,lamda)
           gradient=-1/batch_size*np.dot((y-y_prob[i*batch_size:(i+1)*batch_size]).T,x)+lamda*weight
           weight-=alpha*gradient
           print('loss:'+str(loss))
           Iter+=1
           i+=1
        self.weight=weight   
        return self.weight    
    
    def predict(self,x):
        b = np.ones((x.shape[0], 1))
        x = np.hstack((x, b))
        return np.argmax(np.dot(x,self.weight.T),axis=1)
    
def get_accuracy(label,y):
    count=0
    for i in range(len(label)):
        if label[i]==y[i]:
            count+=1
    return count/len(label)



if __name__=='__main__':
    x_train=load_data(list(train['Phrase']))
    y_train=list(train['Sentiment'])
    loss_list=[]
    clf=SoftmaxRegression()
    clf.weight,loss_list=clf.bgd(x_train,y_train,270,1,0.001)
    x_dev=load_data(list(dev['Phrase']))
    y_dev=list(dev['Sentiment'])
    y_pred=list(clf.predict(x_dev))
    print('accuracy:'+str(get_accuracy(y_dev,y_pred)))
    
    x_test=load_data(list(test['Phrase']))
    y_test=list(clf.predict(x_test))

#    from sklearn.linear_model import LogisticRegression
#    lg = LogisticRegression(C=5)
#    lg.fit(x_train, y_train)
#    print(lg.score(x_dev, y_dev))
    
#    from sklearn.ensemble import RandomForestClassifier
#    forest = RandomForestClassifier( n_estimators = 100, n_jobs=2)
#    forest.fit(x_train,y_train)
#    print(forest.score(x_dev,y_dev))
    
    
    
#    learning_rate=[0.001,0.01,0.1,1,10]
#    loss_all=[]
#    loss_all.append(loss_list)
#    from matplotlib import pyplot as plt 
#    x=list(range(2000))
#    plt.plot(x,loss_all[0],label='0.001')
#    plt.plot(x,loss_all[1],label='0.01')
#    plt.plot(x,loss_all[2],label='0.1')
#    plt.plot(x,loss_all[3],label='1')
#    plt.plot(x,loss_all[4],label='10')
#    plt.xlabel('iter')
#    plt.ylabel('loss')
#    plt.legend(bbox_to_anchor=(1, 1))
#    plt.show()
    
#    lamda=[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]
#    acc_tr=[]
#    acc_dev=[]
#    for l in lamda:
#        clf.weight,loss_list=clf.bgd(x_train,y_train,270,1,l)
#        y_pred=list(clf.predict(x_dev))
#        acc_dev.append(round(get_accuracy(y_dev,y_pred),4))
#        y_pred=list(clf.predict(x_train))
#        acc_tr.append(round(get_accuracy(y_train,y_pred),4))
#    
#    plt.plot(lamda,acc_tr,label='Acc_train')
#    plt.plot(lamda,acc_dev,label='ACC_dev')
#    plt.xlabel('lamda')
#    plt.ylabel('Accuracy')
#    plt.legend()
#    plt.show()
    
#    import time
#    t1=time.clock()
#    loss_list=[]
#    clf=SoftmaxRegression()
#    clf.weight=clf.mgd(x_train,y_train,10,270,0.5,0.01)
#    y_pred=list(clf.predict(x_dev))
#    print('accuracy:'+str(get_accuracy(y_dev,y_pred)))
#    print(time.clock()-t1)
#    