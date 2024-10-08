
# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config as C

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'FastText'
        self.train_path = dataset + C.train_data                                # 训练集
        #self.dev_path = dataset + '/data/dev.txt'                              # 验证集
        self.test_path = dataset + C.test_data                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = C.dropout                                                         # 随机失活
        self.require_improvement = C.require_improvement                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = C.num_classes                                                 # 类别数
        self.n_vocab = C.n_vocab                                                         # 词表大小，在运行时赋值
        self.num_epochs = C.num_epochs                                                   # epoch数
        self.batch_size = C.batch_size                                                   # mini-batch大小
        self.pad_size = C.pad_size                                                       # 每句话处理成的长度(短填长切)  补齐
        self.learning_rate = C.learning_rate                                             # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else C.embedding_size           # 字向量维度
        self.hidden_size = 256                                                       # 隐藏层大小
        self.n_gram_vocab = 250499                                                   # ngram 词表大小


'''Bag of Tricks for Efficient Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.embedding_ngram2 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.embedding_ngram3 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed * 3, config.hidden_size)
        # self.dropout2 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):

        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
