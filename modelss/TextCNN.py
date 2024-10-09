# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config as C


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
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
            if self.embedding_pretrained is not None else C.embedding_size               # 字向量维度
        self.filter_sizes = C.filter_sizes                                               # 卷积核尺寸
        self.num_filters = C.num_filters                                                 # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
            #print(self.embedding)
            #exit()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        #print(out.size())
        #exit()
        out = out.unsqueeze(1)
        #print(print(out.size()))
        #exit()
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        #print(out.size())
        #exit()
        return out

