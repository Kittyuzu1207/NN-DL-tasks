# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:24:14 2019

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

class Multi_TextCNN(nn.Module):
    def __init__(self,args):
        super(Multi_TextCNN, self).__init__()
        self.args = args
        
        self.embedding_dim = args['embedding_dim']
        self.vocab_size = args['vocab_size']
        self.filter_sizes = args['filter_sizes']
        self.filter_num = args['filter_num'] # 卷积核的个数
        self.label_num=args['label_num']  #标签个数
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding_static = self.embedding.from_pretrained(args['vectors'], freeze=not args['fine_tune'])

        self.convs = nn.ModuleList(
           [nn.Conv2d(2, self.embedding_dim, (K, self.embedding_dim), padding=(K // 2, 0)) for K in self.filter_sizes])

        self.linear = nn.Linear(2*len(self.filter_sizes)*self.filter_num, self.label_num)
        self.embed_dropout = nn.Dropout(args['embed_dropout'])
        self.fc_dropout = nn.Dropout(args['fc_dropout'])


    def forward(self,x):  #x是索引的list
        x1 = self.embedding_static(x)  # torch.Size([100, 10, 32])
        x1 = torch.tensor(x1,dtype=torch.float32)  #[100,10,32]
        x2 = self.embedding(x)    #[100,10,32]
        x = torch.stack([x1, x2], 1)  #[100,2,10,32]
        out = self.embed_dropout(x)  #[100, 2, 10, 32]
        l = []
        for conv in self.convs:
           l.append(F.relu(conv(out)).squeeze(3))  #[100,32,10]
        out = l
        l = []
        for i in out:
            l.append(F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2))  #[100,32]
        out = torch.cat(l, 1)    #[100, 96]
        out = self.fc_dropout(out)
        out = self.linear(out)  
        return out
  
    
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]  
    return stopwords 
stoplist=stopwordslist('E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/5基于机器学习的文本分类/stopword.txt')

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
    'embedding_dim': 32,
    'label_num': 0,
    "filter_num": 16,
    "filter_sizes": [3, 4, 5],
    "embed_dropout": 0.5,
    "fc_dropout":0.5,
    'lr':0.001,
    'epoch':10,
    'batch_size':500
}

def train():
    args['label_num']= len(np.unique(y_train))
    print('init net...')
    net = Multi_TextCNN(args)
    print('load embeddings...') 
    train_index=torch.LongTensor(word2index(word_vocab,x_train,10,word_to_idx))
    val_index=torch.LongTensor(word2index(word_vocab,x_dev,10,word_to_idx))
    #输入的是索引，然后用内置函数embedding
    optimizer = torch.optim.Adam(net.parameters(), lr=net.args['lr'])
    target=Variable(torch.LongTensor((np.array(y_train))))
    torch_dataset = Data.TensorDataset(train_index,target) #优化batch和清空中间变量以腾出内存
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
            if step % 10==0:
                with torch.no_grad():
                    logits=net(train_index)
                corrects = (torch.max(logits, 1)[1] == target).sum()
                train_acc = 100* float(corrects.item() / target.size(0))
                print('train accuracy:'+"{:.2f}".format(train_acc)+'%')
                train_acc_list.append("{:.2f}".format(train_acc)+'%')
                dev_acc=dev_val(val_index,y_dev,net)
                dev_acc_list.append("{:.2f}".format(dev_acc)+'%')
    return loss_list,train_acc_list,dev_acc_list
            