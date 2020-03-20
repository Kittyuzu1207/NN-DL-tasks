# NLP beginner 上手教程 [https://github.com/FudanNLP/nlp-beginner]
## 任务一：基于机器学习的文本分类
  1.实现基于logistic/softmax regression的文本分类
  2.实现要求：NumPy
  3.需要了解的知识点：
    文本特征表示：Bag-of-Word，N-gram
    分类器：logistic/softmax regression，损失函数、（随机）梯度下降、特征选择
    数据集：训练集/验证集/测试集的划分
  4.实验：
    分析不同的特征、损失函数、学习率对最终分类性能的影响
    shuffle 、batch、mini-batch

## 任务二：基于深度学习的文本分类
  1.熟悉Pytorch，用Pytorch重写《任务一》，实现CNN、RNN的文本分类；
  2.word embedding 的方式初始化/随机embedding的初始化方式/用glove 预训练的embedding进行初始化
  3. CNN/RNN的特征抽取/词嵌入/Dropout
  
## 任务三：基于注意力机制的文本匹配
  1.输入两个句子判断，判断它们之间的关系。参考ESIM（可以只用LSTM，忽略Tree-LSTM），用双向的注意力机制实现。
  2.数据集：https://nlp.stanford.edu/projects/snli/
  3.Pytorch
  4.注意力机制
    token2token attetnion
    
## 任务四：基于LSTM+CRF的序列标注
  1.用LSTM+CRF来训练序列标注模型：以Named Entity Recognition为例。
  2.数据集：CONLL 2003
  3.Pytorch
  4.评价指标：precision、recall、F1
    无向图模型、CRF

## 任务五：基于神经网络的语言模型
  用LSTM、GRU来训练字符级的语言模型，计算困惑度
  语言模型：困惑度等
  文本生成
