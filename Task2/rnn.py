# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 18:22:08 2019

@author: Xinyi Mou
"""

#construction of RNN
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import embedding
import torch.utils.data as Data 
torch.manual_seed(1)


#load data
word_vocab=[]
word_vocab.append('')
for i in range(len(x_train)):
    word_vocab+=x_train[i]
word_vocab=set(word_vocab)
vocab_size=len(word_vocab)
#词表到索引的映射

def word2index(word_vocab,x_train,max_len):
    word_to_idx={word:i for i,word in enumerate(word_vocab)}
    idx_to_word={word_to_idx[word]:word for word in word_to_idx}
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


#model
class TextRNN(nn.Module):
    def __init__(self, args):
        super(TextRNN, self).__init__()
        embedding_dim=args['embedding_dim']
        vocab_size=args['vocab_size']
        label_num = args['label_num']   
        self.hidden_size=args['hidden_size']  #隐层单元数
        self.layer_num=args['layer_num']   #隐层数
        self.bidirectional = args['bidirectional']  #是否使用双向
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if args['static']:
             self.embedding = self.embedding.from_pretrained(args['vectors'], freeze=not args['fine_tune'])
        
        self.lstm=nn.LSTM(embedding_dim,self.hidden_size,self.layer_num,batch_first=True,bidirectional=self.bidirectional)

        self.fc = nn.Linear(self.hidden_size * 2, label_num) if self.bidirectional else nn.Linear(self.hidden_size, label_num)
        
    def forward(self,x):
        #隐层初始化
        if self.bidirectional:
            h0=torch.zeros(self.layer_num*2,x.size(0),self.hidden_size)
            c0=torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size)
        else:
            h0=torch.zeros(self.layer_num, x.size(0), self.hidden_size)
            c0=torch.zeros(self.layer_num, x.size(0), self.hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))  #hn，cn取最后一个状态
        out = self.fc(out[:, -1, :])   
        return out
    
    
        
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


#using random embedding
    
args={
          'embedding_dim':32,
          'vocab_size':vocab_size,
          'label_num':len(np.unique(y_train)),
          'hidden_size':64,
          'layer_num':1,
          'bidirectional':True,
          'vectors':None,
          'static':False,
          'fine_tune':True,
          'lr':0.001,
          'epoch':10,
          'batch_size':500
          }
def train_random(args,x_train,y_train,x_dev,y_dev):
    print('init net...')
    net=TextRNN(args)
    random_emb_index=word2index(word_vocab,x_train,10)
    train_vec=net.embedding(torch.LongTensor(random_emb_index))
    dev_index=word2index(word_vocab,x_dev,5)
    val_vec=net.embedding(torch.LongTensor(dev_index))
    target=Variable(torch.LongTensor((np.array(y_train))))
    optimizer = torch.optim.Adam(net.parameters(), lr=args['lr'])
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
            if step % 10==0:
                with torch.no_grad():
                    logits=net(train_vec)
                corrects = (torch.max(logits, 1)[1] == target).sum()
                train_acc = 100* float(corrects.item() / target.size(0))
                print('train accuracy:'+"{:.2f}".format(train_acc)+'%')
                dev_acc=dev_val(val_vec,y_dev,net)
                train_acc_list.append("{:.2f}".format(train_acc)+'%')
                dev_acc_list.append("{:.2f}".format(dev_acc)+'%')
    return loss_list,train_acc_list,dev_acc_list
    

#using pretrained embedding  

pretrained_vec,model=embedding.pretrain_w2v(x_train)
#pretrained_vec,model=embedding.pretrain_glove()
#pretrained_vec,model=embedding.pretrain_fasttext(x_train)
word_vocab=model.wv.index2word
word_vocab.append('')
pretrained_vec=np.vstack((pretrained_vec,np.zeros((1,len(pretrained_vec[0])))))
vocab_size=len(word_vocab)
pretrained_vec=torch.from_numpy(pretrained_vec)

args={
          'embedding_dim':96,
          'vocab_size':vocab_size,
          'label_num':len(np.unique(y_train)),
          'hidden_size':64,
          'layer_num':1,
          'bidirectional':True,
          'vectors':pretrained_vec,
          'static':True,
          'fine_tune':False,
          'lr':0.001,
          'epoch':10,
          'batch_size':512
          }    

def train(args,pretrained_vec,x_train,y_train,x_dev,y_dev):  #pretrained_vec 是预训练好的词向量表
    print('init net...')
    net=TextRNN(args)
    train_index=word2index(word_vocab,x_train,10)
    train_vec=net.embedding(torch.LongTensor(train_index))
    train_vec= torch.tensor(train_vec, dtype=torch.float32)
    val_index=word2index(word_vocab,x_dev,10)
    val_vec=net.embedding(torch.LongTensor(val_index))
    val_vec= torch.tensor(val_vec, dtype=torch.float32)
    target=Variable(torch.LongTensor((np.array(y_train))))
    optimizer = torch.optim.Adam(net.parameters(), lr=args['lr'])
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
            if step % 200==0:
                with torch.no_grad():
                    logits=net(train_vec)
                corrects = (torch.max(logits, 1)[1] == target).sum()
                train_acc = 100* float(corrects.item() / target.size(0))
                print('train accuracy:'+"{:.2f}".format(train_acc)+'%')
                dev_acc=dev_val(val_vec,y_dev,net)
                train_acc_list.append("{:.2f}".format(train_acc)+'%')
                dev_acc_list.append("{:.2f}".format(dev_acc)+'%')
    return loss_list,train_acc_list,dev_acc_list
    
    
  
