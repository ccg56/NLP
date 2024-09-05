

#参数配置


chose_model = 'DPCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer




#数据集路径配置：
train_data = '/data/train_data'                     #在文件夹THUCNews中data文件夹下
test_data = '/data/test_data'                       #在文件夹THUCNews中data文件夹下




#超参数
filter_sizes = [2,3,4,5,6]  #TextCNN 卷积核大小


dropout = 0.5                                              # 随机失活
require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
num_classes = 2                                            # 类别数
n_vocab = 0                                                # 词表大小，在运行时赋值
num_epochs = 100                                           # epoch数
batch_size = 64                                           # mini-batch大小
pad_size = 80                                            # 每句话处理成的长度(短填长切)  补齐
learning_rate = 1e-3                                       # 学习率


num_filters = 256                                          # 通道数
embedding_size = 300                                       #embedding向量数

#textrnn