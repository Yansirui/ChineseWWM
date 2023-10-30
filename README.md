# ChineseWWM
有关中文版本的全词掩码代码(离线处理数据版本)  
Random_and_Whole_data_gen.py : 对初始文本进行掩码，指定中文词表，利用该词表对文本进行掩码，掩码率为0.15，如果词表掩码低于0.15就选择使用随机掩码：词掩码为'♥'，随机掩码token为'☕'  
Processor.py: 该文件直接针对于离线的掩码策略，即通过对当前token的值去决定是否掩码，不是掩码的label设置为-100(根据transformers的processor编写)  
Train.py：训练代码  
Random_mask.py：直接用transformers实现的随机掩码，即非离线的随即掩码
ps：如果想要实现离线的随机掩码，将terms列表设置为空[]即可!  
