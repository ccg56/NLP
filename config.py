

#��������


chose_model = 'TextCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer




#���ݼ�·�����ã�
train_data = '/data/train_data'                     #���ļ���THUCNews��data�ļ�����
test_data = '/data/test_data'                       #���ļ���THUCNews��data�ļ�����




#������
filter_sizes = [2,3,4,5,6]  #TextCNN ����˴�С


dropout = 0.5                                              # ���ʧ��
require_improvement = 1000                                 # ������1000batchЧ����û����������ǰ����ѵ��
num_classes = 2                                            # �����
n_vocab = 0                                                # �ʱ��С��������ʱ��ֵ
num_epochs = 100                                           # epoch��
batch_size = 78                                            # mini-batch��С
pad_size = 61                                            # ÿ�仰����ɵĳ���(�����)  ����
learning_rate = 1e-3                                       # ѧϰ��


num_filters = 256                                          # ͨ����
embedding_size = 300                                       #embedding������
