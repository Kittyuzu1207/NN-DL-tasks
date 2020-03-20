# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 21:51:26 2019

@author: Xinyi Mou
"""

import numpy as np
import embedding

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data 
torch.manual_seed(1)

def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]  
    return stopwords 
stoplist=stopwordslist('E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/5基于机器学习的文本分类/stopword.txt')

#def sen2vec(sen_list,model,embedding_dim):
#    if len(sen_list)>0:
#        if sen_list[0] in model.wv.index2word:
#            tmp=model.wv[sen_list[0]]
#        else:
#            tmp=np.zeros((1,embedding_dim))
#        for i in range(1,len(sen_list)):
#            if sen_list[i] in model.wv.index2word:
#                tmp=np.vstack((tmp,model.wv[sen_list[i]]))
#            else:
#                tmp=np.vstack((tmp,np.zeros((1,embedding_dim))))
#        return tmp
#
#
#def load_data(x_train,max_len,embedding_dim,model):
#    #加载train和val的词向量  
#    train_vec=torch.zeros(len(x_train),max_len,embedding_dim)
#    for i in range(len(x_train)):
#        if len(x_train[i])>max_len:
#            tmp=sen2vec(x_train[i][:max_len],model,embedding_dim)
#            train_vec[i]=torch.from_numpy(tmp)
#        elif len(x_train[i])>0:
#            tmp=sen2vec(x_train[i],model,embedding_dim)
#            tmp=np.vstack((tmp,np.zeros((max_len-len(x_train[i]),embedding_dim))))
#            train_vec[i]=torch.from_numpy(tmp)
#        else:
#            train_vec[i]=torch.zeros((max_len,embedding_dim))
#    return train_vec
#对于w2v和glove，词向量是static的，训练过程不更新
#对于随机的来说，词向量是not static的，训练过程中调整词向量


#cnn construction


class TextCNN(nn.Module):
    def __init__(self,args):
        super(TextCNN, self).__init__()
        self.args = args
        
        label_num=args['label_num']  #标签个数
        filter_num = args['filter_num'] # 卷积核的个数
        filter_sizes = args['filter_sizes']
        
        vocab_size = args['vocab_size']
        embedding_dim = args['embedding_dim']
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if args['static']: # 如果使用预训练词向量，则提前加载，当不需要微调时设置freeze为True
            self.embedding = self.embedding.from_pretrained(args['vectors'], freeze=not args['fine_tune'])

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, embedding_dim)) for fsz in filter_sizes])
        self.dropout = nn.Dropout(args['dropout'])
        self.linear = nn.Linear(len(filter_sizes)*filter_num, label_num)

    def forward(self,x):
        #输入x的维度为(batch_size, max_len,embedding_dim)
        # 经过view函数x的维度变为(batch_size, input_chanel=1, w=max_len, h=embedding_dim)
        x = x.view(x.size(0), 1, x.size(1), self.args['embedding_dim']) 
        # 经过卷积运算,x中每个运算结果维度为(batch_size, out_chanel, w, h=1)
        x = [F.relu(conv(x)) for conv in self.convs]
        # 经过最大池化层,维度变为(batch_size, out_chanel, w=1, h=1)
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        # 将不同卷积核运算结果维度（batch，out_chanel,w,h=1）展平为（batch, outchanel*w*h）
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        # 将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
        x = torch.cat(x, 1)
        # dropout层
        x = self.dropout(x)
        # 全连接层
        logits = self.linear(x)
        return logits

    
def dev_val(val_vec,y_dev,net):
    corrects=0
    target=Variable(torch.LongTensor((np.array(y_dev))))
    with torch.no_grad():
        logits=net(val_vec)
    corrects = (torch.max(logits, 1)
                     [1].view(target.size()) == target).sum()
    accuracy = 100.0 *float(corrects.item()/val_vec.size(0))
    print('val accuracy:'+"{:.2f}".format(accuracy)+'%')
    return accuracy


'''for pretrained embedding'''
pretrained_vec,model=embedding.pretrain_w2v(x_train)
#pretrained_vec,model=embedding.pretrain_glove()
#pretrained_vec,model=embedding.pretrain_fasttext(x_train)
word_vocab=model.wv.index2word
word_vocab.append('')
pretrained_vec=np.vstack((pretrained_vec,np.zeros((1,len(pretrained_vec[0])))))
word_to_idx={word:i for i,word in enumerate(word_vocab)}
idx_to_word={word_to_idx[word]:word for word in word_to_idx}
#去停用词
del_list=[]
for w in word_vocab:
    if w in stoplist:
        word_vocab.remove(w)
        del_list.append(word_to_idx[w])
pretrained_vec=np.delete(pretrained_vec,del_list, axis = 0)
word_to_idx={word:i for i,word in enumerate(word_vocab)}
idx_to_word={word_to_idx[word]:word for word in word_to_idx}
vocab_size=len(word_vocab)
pretrained_vec=torch.from_numpy(pretrained_vec)

##生成训练数据，将训练的Word转换为word的索引
def word2index(word_vocab,x_train,max_len,word_to_idx):
    texts_with_id=np.zeros((len(x_train),max_len))
    for i in range(len(x_train)):
        if(len(x_train[i])>0):
            if len(x_train[i])<max_len:
                for j in range(len(x_train[i])):
                    if x_train[i][j] in word_vocab:
                        texts_with_id[i][j]=word_to_idx[x_train[i][j]]
                    else:
                        texts_with_id[i][j] = word_to_idx['']
                for j in range(len(x_train[i]),max_len):
                    texts_with_id[i][j] = word_to_idx['']
            else:
                for j in range(max_len):
                    if x_train[i][j] in word_vocab:
                        texts_with_id[i][j]=word_to_idx[x_train[i][j]]
                    else:
                        texts_with_id[i][j] = word_to_idx['']
        else:
            for j in range(max_len):
                texts_with_id[i][j] = word_to_idx['']
    return texts_with_id


args = {
    'vectors':pretrained_vec,  #预训练的词表
    'static':True,
    'fine_tune':False,
    'vocab_size': vocab_size,
    'embedding_dim': 96,
    'label_num': 0,
    "filter_num": 16,
    "filter_sizes": [3, 4, 5],
    "dropout": 0.5,
    'lr':0.001,
    'epoch':10,
    'batch_size':512
}
    
def train(args,x_train,y_train,x_dev,y_dev,max_len):
    args['label_num']= len(np.unique(y_train))
    print('init net...')
    net = TextCNN(args)
    print('load embeddings...') 
    train_index=word2index(word_vocab,x_train,10,word_to_idx)
    train_vec=net.embedding(torch.LongTensor(train_index))
    train_vec=torch.tensor(train_vec,dtype=torch.float32)
    val_index=word2index(word_vocab,x_dev,10,word_to_idx)
    val_vec=net.embedding(torch.LongTensor(val_index))
    val_vec=torch.tensor(val_vec,dtype=torch.float32)
    #输入的是索引，然后用内置函数embedding
    optimizer = torch.optim.Adam(net.parameters(), lr=net.args['lr'])
    target=Variable(torch.LongTensor((np.array(y_train))))
    torch_dataset = Data.TensorDataset(train_vec,target) #优化batch和清空中间变量以腾出内存
    loader = Data.DataLoader( 
    dataset=torch_dataset, 
    batch_size=args['batch_size'], 
    shuffle=True
    ) 
    loss_list=[]
    train_acc_list=[]
    dev_acc_list=[]
    for i in range(args['epoch']):
        print('iter'+str(i)+' training')
        for step, (batch_x, batch_y) in enumerate(loader): 
            print('step:'+str(step))
            b_x = Variable(batch_x) 
            b_y = Variable(batch_y)
            optimizer.zero_grad()
            logits=net(b_x)
            loss=F.cross_entropy(logits,b_y)
            print('loss:'+str(loss.item()))
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            if step % 100 ==0:
                with torch.no_grad():
                    logits=net(train_vec)
                corrects = (torch.max(logits, 1)[1] == target).sum()
                train_acc = 100* float(corrects.item() / target.size(0))
                print('train accuracy:'+"{:.2f}".format(train_acc)+'%')
                train_acc_list.append("{:.2f}".format(train_acc)+'%')
                dev_acc=dev_val(val_vec,y_dev,net)
                dev_acc_list.append("{:.2f}".format(dev_acc)+'%')
    return loss_list,train_acc_list,dev_acc_list



'''for random embedding'''

word_vocab=[]
word_vocab.append('')
for i in range(len(x_train)):
    word_vocab+=x_train[i]
word_vocab=set(word_vocab)-set(stoplist)
vocab_size=len(word_vocab)
#词表到索引的映射
word_to_idx={word:i for i,word in enumerate(word_vocab)}
idx_to_word={word_to_idx[word]:word for word in word_to_idx}



args = {
    'vectors':None,  
    'static':False,
    'fine_tune':True,
    'vocab_size': vocab_size,
    'embedding_dim': 32,
    'label_num':0,
    "filter_num": 16,
    "filter_sizes": [3, 4, 5],
    "dropout": 0.5,
    'lr':0.0001,
    'epoch':10,
    'batch_size':500
} 
def train_random(args,x_train,y_train,x_dev,y_dev,max_len):
    print('init net...')
    args['label_num']= len(np.unique(y_train))
    net = TextCNN(args)
    #对于随机embedding，输入的是索引，然后用内置函数embedding
    random_emb_index=word2index(word_vocab,x_train,10,word_to_idx)
    train_vec=net.embedding(torch.LongTensor(random_emb_index))
    dev_index=word2index(word_vocab,x_dev,10,word_to_idx)
    val_vec=net.embedding(torch.LongTensor(dev_index))
    optimizer = torch.optim.Adam(net.parameters(), lr=net.args['lr'])
    target=Variable(torch.LongTensor((np.array(y_train))))
    torch_dataset = Data.TensorDataset(train_vec,target) #优化batch和清空中间变量以腾出内存
    loader = Data.DataLoader( 
    dataset=torch_dataset, 
    batch_size=args['batch_size'], 
    shuffle=True
    ) 
    loss_list=[]
    train_acc_list=[]
    dev_acc_list=[]
    for i in range(args['epoch']):
        print('iter'+str(i)+' training')
        for step, (batch_x, batch_y) in enumerate(loader): 
            print('step:'+str(step))
            b_x = Variable(batch_x) 
            b_y = Variable(batch_y)
            optimizer.zero_grad()
            logits=net(b_x)
            loss=F.cross_entropy(logits,b_y)
            print('loss:'+str(loss.item()))
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            if step % 10 ==0:
                with torch.no_grad():
                    logits=net(train_vec)
                corrects = (torch.max(logits, 1)[1] == target).sum()
                train_acc = 100* float(corrects.item() / target.size(0))
                print('train accuracy:'+"{:.2f}".format(train_acc)+'%')
                train_acc_list.append("{:.2f}".format(train_acc)+'%')
                dev_acc=dev_val(val_vec,y_dev,net)
                dev_acc_list.append("{:.2f}".format(dev_acc)+'%')            
    return loss_list,train_acc_list,dev_acc_list


import matplotlib.pyplot as plt 
plt.plot(list(range(len(loss_list))), loss_list,label='loss') 
plt.xlabel('Steps') 
plt.ylabel('Loss') 
plt.legend(loc='best')
plt.show()   
  
plt.plot(list(range(len(train_acc_list))),train_acc_list,label='train_acc')
plt.plot(list(range(len(dev_acc_list))),dev_acc_list,label='dev_acc')
plt.legend(loc='best')
plt.show() 