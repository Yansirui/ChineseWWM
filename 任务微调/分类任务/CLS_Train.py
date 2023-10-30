import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM, AdamW,BertTokenizerFast
import json
import torch
from transformers import BertTokenizer,BertTokenizerFast
from torch.utils.data import Dataset
from typing import List, Dict
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_scheduler
from model_class import *
# 读取JSON文件
data_file = "/home/sirui/WMM/Car/Data/mean_train.json"
def read_json(json_path):
    with open(json_path,'r',encoding='utf-8') as f:
        str=f.read()
        data=json.loads(str)
        return data
# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_file, tokenizer):
        self.data = read_json(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample= {
                'sen1':self.data[index]['question'],
                'cls_label':self.data[index]['label']
                }
        return sample


def collate_fn(batch: List[Dict[str, any]],tokenizer=BertTokenizerFast.from_pretrained(r'/home/sirui/car_corpus/RoBERTa')) -> Dict[str, torch.Tensor]:
    sentences1=[example["sen1"] for example in batch]
    #sentences2=[example["sen2"] for example in batch]
    labels=[example["cls_label"] for example in batch]
    tokens = tokenizer(sentences1,truncation=True,padding=True,return_tensors='pt')
    tokens['labels']=torch.tensor(labels)
    return tokens
labels=[]
for con in read_json(data_file):
    if con['label'] not in labels:
        labels.append(con['label'])
import torch
import random
import numpy as np

# 设置随机种子以确保结果的可重现性
import torch
import random
import numpy as np


# tokenizer
model_name='Test'
check_point = '/home/sirui/WMM/Law/model/BERT_Model/BERT_BASE'
print(len(labels))
model = ClassificationModel(num_labels=len(labels))
tokenizer = BertTokenizer.from_pretrained(check_point)

# 加载数据集
train_dataset = CustomDataset(data_file, tokenizer)

# 定义超参数
batch_size = 32
num_epochs = 3
learning_rate = 5e-5

# 初始化优化器和损失函数
optimizer = AdamW(model.parameters(), lr=learning_rate)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# 将模型移动到GPU（如果可用）
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# 创建数据加载器

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
progress_bar = tqdm(range(num_training_steps))
# 训练循环
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs=model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'],  cls_labels=batch['labels'])
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        progress_bar.set_description_str(f'Epoch{epoch}')
        progress_bar.set_postfix(loss=loss.item())
    # 保存训练后的模型
save_directory = "/home/sirui/WMM/Car/model/Forcls/"+model_name
model.save_pretrained(save_directory)

