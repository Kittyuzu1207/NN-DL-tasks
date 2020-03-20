# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 22:25:35 2020

@author: Xinyi Mou
"""

#use_crf=False,only use argmax to output of lstm

from train2 import *


def eval(model, loss, data_iterator):
    acc = 0
    l = 0
    c = 0
    p=0
    r=0
    f=0
    with torch.no_grad():
        for sen,word,mask,sen_len,label in data_iterator:
            logits = net(sen,word,sen_len)
            logits_tmp=[]
            label_tmp=[]
            for i in range(len(sen_len)):
                logits_tmp.append(logits[i][:sen_len[i]])
                label_tmp.append(label[i][:sen_len[i]])
            logits=torch.cat(logits_tmp,dim=0)
            label=torch.cat(label_tmp,dim=0)
            logits=logits.view(-1,9)
            label=label.view(-1,1).squeeze()
            batch_l = loss(logits, label).item()
            batch_a = int(torch.sum(torch.argmax(logits, dim=1) == label))/int(label.shape[0])            
            p1,r1,f1=PRF(label,torch.argmax(logits, dim=1))
            p+=p1
            r+=r1
            f+=f1
            c += 1
            l += batch_l
            acc += batch_a
    return acc/c, l/c,p/c,r/c,f/c

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
          'label_num':9,
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
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.015)
    max_epoch = 10
    for epoch in range(max_epoch):
        c=0
        l=0
        acc=0
        net.train()
        for sen,word,mask,sen_len,label in loader:
            optimizer.zero_grad()
            logits = net(sen,word,sen_len)
            logits_tmp=[]
            label_tmp=[]
            for i in range(len(sen_len)):
                logits_tmp.append(logits[i][:sen_len[i]])
                label_tmp.append(label[i][:sen_len[i]])
            logits=torch.cat(logits_tmp,dim=0)
            label=torch.cat(label_tmp,dim=0)
            logits=logits.view(-1,9)
            label=label.view(-1,1).squeeze()
            batch_loss=loss(logits,label)
            print(c,batch_loss.item())
            batch_accu = int(torch.sum(torch.argmax(logits, dim=1) == label))/int(label.shape[0])
            batch_loss.backward()
            optimizer.step()
            
            c+=1
            l+=batch_loss.item()
            acc += batch_accu
            if c % 100 == 0:
                print(c, batch_loss, batch_accu)
                print('P R F:')
                print(PRF(label,torch.argmax(logits, dim=1)))
            
    train_acc,train_loss,p,r,f=eval(net,loss,loader)
    print('accuracy on train:', train_acc, 'loss on train:', train_loss)
    print('P R F:')
    print(p,r,f)
    valid_acc, valid_loss,p,r,f = eval(net, loss,data_iterator=dev_loader)
    print('accuracy on dev:', valid_acc, 'loss on dev:', valid_loss)
    print('P R F:')
    print(p,r,f)
    test_acc, test_loss,p,r,f = eval(net, loss=loss, data_iterator=test_loader)
    print('final accuracy on test:', test_acc, 'loss on test:', test_loss)
    print('P R F:')
    print(p,r,f)
    
def PRF(label,predict):
    label=list(label)
    predict=list(predict)
    TP={'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0}
    TN={'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0}
    FP={'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0}
    FN={'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0}
    precision={}
    recall={}
    for i in range(len(label)):
         for key in TP.keys():
            if predict[i]==int(key) and label[i]==int(key):
                TP[key]+=1
            elif label[i]==int(key) and predict[i]!=int(key):
                FN[key]+=1
            elif label[i]!=int(key) and predict[i]!=int(key):
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