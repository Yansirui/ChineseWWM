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
def read_json(json_path):
    with open(json_path,'r',encoding='utf-8') as f:
        str=f.read()
        data=json.loads(str)
        return data

predicted_labels=[]
true_labels=[]
model_name='Test'
checkpoint = "/home/sirui/WMM/Car/model/Forcls/"+model_name
model=ClassificationModel.from_pretrained(cls=ClassificationModel,pretrained_model_name_or_path=checkpoint,num_labels=10)
tokenizer=AutoTokenizer.from_pretrained(r'/home/sirui/WMM/Finance/model/BERT_Model/BERT_BASE')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
data=read_json('/home/sirui/WMM/Car/Data/mean_test.json')
count=len(data)
texts1=[]
texts2=[]
labels=[]
score=0
for d in data:
    texts1.append(d['question'])
    true_labels.append(d['label'])
print('------------Predicting----------------')
for text1 in texts1:
    #print(q)
    #q=q+1
    input1=tokenizer(text1,truncation=True,padding=True,return_tensors='pt')
    softmax=nn.Softmax(dim=1)
    input1=input1.to(device)
    with torch.no_grad():
        l=softmax(model(**input1)['logits'])[0].tolist()
    index=l.index(max(l))
    predicted_labels.append(index)
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
# ... your code ...
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# ... your code ...

print('------------Validation----------------------')

# Calculate accuracy, precision, recall, and F1-score as before
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro',zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='macro',zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average='macro',zero_division=0)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)


print('---------------end--------------------------')

