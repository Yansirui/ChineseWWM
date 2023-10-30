import torch
#from sentence_transformers import SentenceTransformer
from transformers import AutoModel,AutoTokenizer
import numpy as np
import faiss
import pandas as pd
import json
from Term_CL import TCLBERT
from Sentence_CL import BERTFor_Sencl
#from Model_OnlyLinear import Knowledge_based_Model
def save_csv(file,list):
    pd.DataFrame(list).to_csv(file)
def read_Csv(file_path):
    corpus=pd.read_csv(file_path,index_col=0)
    list1=corpus.values.tolist()
    return list1

def get_questions():
    with open(r'/home/sirui/WMM/Car/Data/Validation_titles.txt','r',encoding='utf-8') as f:
        questions=f.readlines()
   # with open(r'/home/sirui/car_corpus/data/new_questions.txt','r',encoding='utf-8') as f:
   #    questions=f.readlines()
    q=[]
    for question in questions:
        q.append(question.replace('\n',''))
    return q

def read_json(json_path):
    with open(json_path,'r',encoding='utf-8') as f:
        str=f.read()
        data=json.loads(str)
        return data

def save_vectorspace(model_name,text_list):
    #model = AutoModel.from_pretrained(model_name)
    #model=BERTFor_Sencl.from_pretrained(cls=BERTFor_Sencl,pretrained_model_name_or_path=model_name,check_point=r'/home/sirui/WMM/Car/model/Encoder/BERT_BASE').bert
    model=TCLBERT.from_pretrained(cls=TCLBERT,pretrained_model_name_or_path=model_name,check_point=r'/home/sirui/WMM/Car/model/Encoder/BERT_BASE').bert
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vector=[]
    print('tokeniezer')
    encoded_input=tokenizer(text_list,padding=True,truncation=True,return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    print('save begin')
    for i in range(len(text_list)):
        vector.append(model_output.last_hidden_state[i][0].detach().numpy().tolist())
    save_csv(model_name+'/vector_space.csv',vector)
all_index=[]
model_check_point= r"/home/sirui/WMM/Car/model/Term_CL/TCLBERT_BASE-bs=8-lr=3e-05"
Save=False
if Save:
    save_vectorspace(model_check_point,get_questions())
else:
    #model=AutoModel.from_pretrained(model_check_point)
    #model=BERTFor_Sencl.from_pretrained(cls=BERTFor_Sencl,pretrained_model_name_or_path=model_check_point,check_point=r'/home/sirui/WMM/Car/model/Encoder/BERT_BASE').bert
    model=TCLBERT.from_pretrained(cls=TCLBERT,pretrained_model_name_or_path=model_check_point,check_point=r'/home/sirui/WMM/Car/model/Encoder/BERT_BASE').bert
    tokenizer=AutoTokenizer.from_pretrained(model_check_point)
    #100个测试题目，400个改写，每个题目在向量空间中找出最像的20个题目，测试题目排在第i个，得分为（20-i）/20，之后再乘以1/400
    score=0
    MRR=0
    #embeddings=read_Csv('Sentence_model/384-d_vector_space.csv')
    embeddings=read_Csv(model_check_point+'/vector_space.csv')
    embeddings=np.array(embeddings).astype('float32')
    index=faiss.IndexFlatL2(768)
    index.add(embeddings)
    i=1
    list1=[]
    #question_path=r"/home/sirui/WMM/Knowledge_based_WMM/validation/Validation.json"
    question_path=r"/home/sirui/WMM/Car/Data/Validation_titles.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device="cpu"
    model=model.to(device)
    questions=read_json(question_path)
    length=len(questions)
    print(length)
    top1=0
    top3=0
    top5=0
    for ques in questions:
        question=ques['title']
        test=ques['question']
        #question=ques['Title']
        #test=ques['Similar_Title']
        encoded_input=tokenizer(test,padding=True,truncation=True,return_tensors='pt')
        encoded_input=encoded_input.to(device)
        with torch.no_grad():
            model_output=model(**encoded_input)
        for i in range(1):
            is_find=False
            #print(model_output.last_hidden_state.size())
            test_emb=model_output.last_hidden_state[i][0].cpu().detach().numpy().astype('float32')
            t=[]
            t.append(test_emb)
            t=np.array(t).astype('float32')
            D,I=index.search(t,40000)
            index1=I[0]
            questions1=get_questions()
            q_i=0
            for i in index1:
                if questions1[i]==question:
                    true_index=q_i
                    is_find=True
                q_i=q_i+1
            all_index.append(true_index)
            if is_find==True:
                if true_index==0:
                    top1=top1+1
                if true_index<=2:
                    top3=top3+1
                if true_index<=4:
                    top5=top5+1
                true_index=float(true_index)
                true_index=true_index+1
                MRR=MRR+1/true_index
    MRR=MRR/length
    print('top1的占比：',top1/length)
    print('top3的占比：',top3/length)
    print('top5的占比：',top5/length)
    print('MRR',MRR)
    #print('LMR:',score)

