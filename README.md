#枯草芽孢杆菌启动子强度预测工具 

## 环境
python 3.7  
pytorch 1.1  
tqdm  
sklearn  
tensorboardX

修改选择的模型：在config.py文件中 “chose_model = 'FastText'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer”替换对应的模型
# 训练并测试：

python run.py 


### 参数
模型都在models目录下，超参定义和模型定义在同一文件中。  
