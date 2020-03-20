# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:22:11 2020

@author: Xinyi MOU
"""

'''
Basic models for LSTM-CRF NER task
CNN_BiLSTM: CNN -- for char encoding 
            LSTM -- use word-char combined embedding to get feats matrix
CRF: Sequence labeling
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from initialize import *
from torch.nn import init
import torch.nn.utils.rnn as rnn_utils


class CNN_BiLSTM(nn.Module):
    def __init__(self,args):
        super(CNN_BiLSTM,self).__init__()
        #basic param
        self.vocab_size=args['vocab_size']
        self.char_size=args['char_size']
        self.embed_dim=args['embed_dim']
        self.char_emb_dim=args['char_emb_dim']
        self.label_num=args['label_num']
        self.paddingId=args['paddingId']
        self.char_paddingId=args['char_paddingId']
        self.pretrained_embed=args['pretrained_embed']
        self.pretrained_weight=args['pretrained_weight']
        self.device=args['device']
        
        #cnn param
        self.conv_filter_sizes=args['conv_filter_sizes']
        self.conv_filter_nums=args['conv_filter_nums']
        #lstm param
        self.hidden_size=args['hidden_size']
        self.num_layers=args['num_layers']
               
        #word embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.paddingId)
        if self.pretrained_embed:
            self.embedding.weight.data.copy_(self.pretrained_weight)
        #char embedding
        self.char_embedding=nn.Embedding(self.char_size,self.char_emb_dim,padding_idx=self.char_paddingId)
        init_embed(self.char_embedding.weight)
        
        self.dropout=nn.Dropout(args['dropout'])
        
        #cnn structure
        self.char_encoders=[]
        for i,filter_size in enumerate(self.conv_filter_sizes):
            f=nn.Conv3d(in_channels=1,out_channels=self.conv_filter_nums[i],kernel_size=(1,filter_size,self.char_emb_dim))
            self.char_encoders.append(f)
        for conv in self.char_encoders:
            if self.device!='cpu':
                conv.cuba()
        
        #lstm structure
        lstm_embed_dim=self.embed_dim+self.char_emb_dim
        self.bilstm=nn.LSTM(input_size=lstm_embed_dim,hidden_size=self.hidden_size,num_layers=self.num_layers,
                            bidirectional=True,batch_first=True,bias=True)
        self.linear=nn.Linear(in_features=self.hidden_size*2,out_features=self.label_num,bias=True)
        init_linear_weight_bias(self.linear)
    
    def char_forward(self,inputs):  #input:[batch_size,sen_len,word_len]  cnn需要补齐长度，lstm用mask解开（char的补齐ok，但seq的补齐要pack）
        sen_len,char_len=inputs.size(1),inputs.size(2)
        inputs=inputs.view(-1,sen_len*char_len)
        input_embed=self.char_embedding(inputs)  #[batch_size,sen_len,word_len,char_emb_dim]
        input_embed=input_embed.view(-1,1,sen_len,char_len,self.char_emb_dim)
        char_conv_outputs=[]
        for encoder in self.char_encoders:
            conv_output=encoder(input_embed)
            pool_output=torch.max(conv_output,-2)[0]
            pool_output=torch.squeeze(pool_output,-1)
            char_conv_outputs.append(pool_output)  #[batch_size,1,sen_len,char_emb_dim]
        char_conv_outputs = torch.cat(char_conv_outputs, dim=1)  #[batch_size,len(char_encoders),sen_len,char_emb_dim]
        char_conv_outputs=char_conv_outputs.permute(0,2,1) #[batch_size,sen_len,len(char_encoders),char_emb_dim]
        return char_conv_outputs
    
    def forward(self,word,char,sen_len):
        encoded_char=self.char_forward(char)
        encoded_char=self.dropout(encoded_char)  #[batch_size,sen_len,char_dim]
        encoded_word=self.embedding(word)  #[batch_size,sen_len,embedding_dim]        
        x = torch.cat((encoded_word, encoded_char), -1) #[batch_size,sen_len,emb_dim]
        sorted_sen_len, sorted_sen_index = sen_len.sort(0, descending=True)
        re_sen, re_sen_index = sorted_sen_index.sort(0, descending=False)
        sorted_sen = x.index_select(0, sorted_sen_index)
        packed_sen = rnn_utils.pack_padded_sequence(sorted_sen, sorted_sen_len, batch_first=True)
        out_sen, _ = self.bilstm(packed_sen, None)
        out_packed_sen, _ = rnn_utils.pad_packed_sequence(out_sen, batch_first=True)
        encoded_sen = out_packed_sen.index_select(0, re_sen_index)
        encoded_sen=self.dropout(encoded_sen)
        encoded_sen=torch.tanh(encoded_sen)
        logits=self.linear(encoded_sen)
#        x, _ = self.bilstm(x)   
#        x = self.dropout(x)
#        x = torch.tanh(x)
#        logits = self.linear(x)
        return logits
        

#lstm 输出的size是tag_size (每个标签对应的概率:词到tag的发射概率值)

#log_sum_exp函数：log_sum_exp(x)=a+log_sum_exp(x-a) a一般取max
def log_sum_exp(vec,tag_size): #vac:[batch_size,tag_size,tag_size]
    _,idx=torch.max(vec,1)
    max_score=torch.gather(vec,1,idx.view(-1,1,tag_size)).view(-1,1,tag_size)
    return max_score.view(-1,tag_size)+torch.log(torch.sum(torch.exp(vec-max_score.expand_as(vec)),1)).view(-1,tag_size)
        

class CRF(nn.Module):
    def __init__(self,args):
        super(CRF,self).__init__()
        self.target_size=args['target_size']
        self.device=args['device']
    
        
        self.START_TAG,self.STOP_TAG=-2,-1
        init_trans=torch.zeros(self.target_size+2,self.target_size+2)   #trans matrix  
        
        init_trans[:, self.START_TAG] = -10000.0
        init_trans[self.STOP_TAG, :] = -10000.0
        self.trans=nn.Parameter(init_trans)  #训练中会更新
        
    def _forward_alg(self,feats,mask):    #前向传播计算score
        #feats:(batch_size,seq_len,self.target_size+2)
        #mask:(batch_size,seq_len)
        batch_size=feats.size(0)
        seq_len=feats.size(1)
        tag_size=feats.size(2)
        mask=mask.transpose(1,0).contiguous()
        ins_num=seq_len*batch_size
        feats=feats.transpose(1,0).contiguous().view(ins_num,1,tag_size).expand(ins_num,tag_size,tag_size)  #为了配合trans的维度
#        trans=torch.empty(self.trans.size())
#        trans=F.softmax(self.trans,dim=1)
   
        scores=feats+self.trans.view(1,tag_size,tag_size).expand(ins_num,tag_size,tag_size)
        scores=scores.view(seq_len,batch_size,tag_size,tag_size)
        #start
        seq_iter=enumerate(scores)
        _,ini_values=next(seq_iter)  #[batch_size,tag_size,tag_size]
        partition=ini_values[:,self.START_TAG,:].clone().view(batch_size,tag_size,1)
        #iter
        for idx,cur_values in seq_iter:
            cur_values=cur_values+partition.contiguous().view(batch_size,tag_size,1).expand(batch_size,tag_size,tag_size)
            cur_partition=log_sum_exp(cur_values,tag_size)
            
            mask_idx=mask[idx,:].view(batch_size,1).expand(batch_size,tag_size)
            masked_cur_partition=cur_partition.masked_select(mask_idx)
            mask_idx=mask_idx.contiguous().view(batch_size,tag_size,1)
            partition.masked_scatter_(mask_idx,masked_cur_partition)
        
        cur_values=self.trans.view(1,tag_size,tag_size).expand(batch_size,tag_size,tag_size)+partition.contiguous().view(batch_size,tag_size,1).expand(batch_size,tag_size,tag_size)
        cur_partition=log_sum_exp(cur_values,tag_size)
        final_partition=cur_partition[:,self.STOP_TAG]
        return final_partition.sum(),scores
    
    def _viterbi_decode(self,feats,mask):  #for prediction  /compute the most likely path and score  动态规划
        #feats:(batch_size,seq_len,self.tag_size+2)
        #mask:(batch_size,seq_len)
        batch_size=feats.size(0)
        seq_len=feats.size(1)
        tag_size=feats.size(2)
        len_mask=torch.sum(mask.long(),1).view(batch_size,1).long() #compute seq_len of each senten
        mask = mask.transpose(1, 0).contiguous()
        ins_sum=batch_size*seq_len
        feats=feats.transpose(1,0).contiguous().view(ins_sum,1,tag_size).expand(ins_sum,tag_size,tag_size)
        scores=feats+self.trans.view(1,tag_size,tag_size).expand(ins_sum,tag_size,tag_size)
        scores=scores.view(seq_len,batch_size,tag_size,tag_size)
        
        seq_iter=enumerate(scores)
        back_points=[]
        partition_history=[]
        mask=(1-mask.long()).byte()
        _,ini_values=next(seq_iter)
        partition = ini_values[:, self.START_TAG, :].clone().view(batch_size, tag_size)
        partition_history.append(partition)
        
        for idx,cur_values in seq_iter:
            cur_values=cur_values+partition.contiguous().view(batch_size,tag_size,1).expand(batch_size,tag_size,tag_size) #(batch_size,tag_size,tag_size)
            partition, cur_bp = torch.max(cur_values, 1) #partition:(batch_size,1,tag_size)  cur_bp:(batch_size,1)
            partition_history.append(partition)
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)
            
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, -1).transpose(1, 0).contiguous() ## (batch_size, seq_len. tag_size)
        last_position = len_mask.view(batch_size,1,1).expand(batch_size, 1, tag_size) -1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size,tag_size,1)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.trans.view(1,tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = torch.zeros(batch_size, tag_size, device=self.device, requires_grad=True).long()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)
        
        pointer = last_bp[:, self.STOP_TAG]
        insert_last = pointer.contiguous().view(batch_size,1,1).expand(batch_size,1, tag_size)
        back_points = back_points.transpose(1,0).contiguous()
        back_points.scatter_(1, last_position, insert_last)
        back_points = back_points.transpose(1,0).contiguous()
        decode_idx = torch.empty(seq_len, batch_size, device=self.device, requires_grad=True).long()
        decode_idx[-1] = pointer.detach()
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.detach().view(batch_size)
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx
        
        
    def _score_sentence(self,scores,mask,tags):  #S(X,y) 根据真实的label计算一个score
        seq_len=scores.size(0)  #scores:(seq_len,batch_size,tag_size,tag_size)
        batch_size=scores.size(1)
        tag_size=scores.size(-1)
        tags=tags.view(batch_size,seq_len)
        #arrange tags to record label bigram information to index
        new_tags = torch.empty(batch_size, seq_len, device=self.device, requires_grad=True).long()
        for idx in range(seq_len):
            if idx ==0:
                new_tags[:,0]=(tag_size-2)*tag_size +tags[:,0]
            else:
                new_tags[:,idx]=tags[:,idx-1]*tag_size+tags[:,idx]
        end_trans=self.trans[:,self.STOP_TAG].contiguous().view(1,tag_size).expand(batch_size,tag_size)
        len_mask=torch.sum(mask,1).view(batch_size, 1).long()
        end_ids=torch.gather(tags,1,len_mask-1)
        
        end_score=torch.gather(end_trans,1,end_ids)
        new_tags=new_tags.transpose(1,0).contiguous().view(seq_len,batch_size,1)
        tag_score=torch.gather(scores.view(seq_len,batch_size,-1),2,new_tags).view(seq_len,batch_size)
        tag_score=tag_score.masked_select(mask.transpose(1,0))
        gold_score = tag_score.sum() + end_score.sum()
    
        return gold_score
    
    def forward(self,feats,mask):
        path_score,best_path=self._viterbi_decode(feats,mask)
        return path_score, best_path
        
    def neg_log_likelihood(self,feats,mask,tags): #object function for loss computation
        batch_size=feats.size(0)
        forward_score,scores=self._forward_alg(feats,mask)
        gold_score=self._score_sentence(scores,mask,tags)
        return forward_score-gold_score
        
 
       
#class Seq_Labeling(nn.Module):
#    def __init__(self,args):
#        self.vocab_size=args['vocab_size']
#        self.embed_dim=args['embed_dim']
#        self.label_num=args['label_num']
#        self.paddingId = args['paddingId']
#        self.dropout = args['dropout']
#        self.lstm_hiddens = args['lstm_hiddens']
#        self.lstm_layers = args['lstm_layers']
#        self.pretrained_embed = args['pretrained_embed']
#        self.pretrained_weight = args['pretrained_weight']
#        self.char_size = args['char_size']
#        self.char_paddingId = args['char_paddingId']
#        self.char_emb_dim = args['char_emb_dim']
#        self.conv_filter_sizes = self._conv_filter(args['conv_filter_sizes'])
#        self.conv_filter_nums = self._conv_filter(args['conv_filter_nums'])
#        assert len(self.conv_filter_sizes) == len(self.conv_filter_nums)
#        self.device = args['device']
#        self.target_size = self.label_num + 2
#        
#        model_args={
#          'vocab_size':self.vocab_size,
#          'char_size':self.char_size,
#          'embed_dim':self.embed_dim,
#          'char_emb_dim':self.char_emb_dim,
#          'label_num':self.label_num,
#          'paddingId':self.paddingId,
#          'char_paddingId':self.char_paddingId,
#          'pretrained_embed':self.pretrained_embed,
#          'pretrained_weight':self.pretrained_weight,
#          'device':self.device,
#          'conv_filter_sizes':self.conv_filter_sizes,
#          'conv_filter_nums':self.conv_filter_nums,
#          'hidden_size':self.hidden_size,  
#          'num_layers':self.num_layers,
#          'dropout':self.dropout                
#                }
#        
#        self.encoder_model = CNN_BiLSTM(model_args)
#        if self.use_crf is True:
#            args_crf = {'target_size': self.label_num, 'device': self.device}
#            self.crf_layer = CRF(args_crf)
#        
#    def _conv_filter(str_list):
#        int_list = []
#        str_list = str_list.split(",")
#        for s in str_list:
#            int_list.append(int(s))
#        return int_list
#    
#    def forward(self,word,char,seq_len):
#        encoder_output = self.encoder_model(word, char, seq_len)
#        return encoder_output