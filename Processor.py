import json
import torch
from transformers import BertTokenizerFast
from torch.utils.data import Dataset
from typing import List, Dict

def read_json(json_path):
    with open(json_path,'r',encoding='utf-8') as f:
        str=f.read()
        data=json.loads(str)
        return data
def read_txt(txt_path):
    with open(txt_path,'r',encoding='utf-8') as f:
        content=f.readlines()
    for i in range(len(content)):
        content[i]=content[i].replace('\n','')
    return content

class Knowledge_Dataset(Dataset):
    def __init__(self, sentence_path,term_path,length_path,word_path):
        self.sentence_path = sentence_path
        self.word_path=word_path
        self.term_path=term_path
        self.length_path=length_path
        self.sentence_corpus = read_txt(self.sentence_path)
        self.words = read_txt(self.word_path)
        self.terms=read_txt(self.term_path)
        self.lengths=read_txt(self.length_path)


    def __len__(self):
        return len(self.sentence_corpus)

    def __getitem__(self, index):
        # 从json中读取数据
        #sentence = self.sentence_corpus
        #tokenizer = BertTokenizer.from_pretrained(r'/home/sirui/car_corpus/RoBERTa')

        # 构造一个样本
        sample = {
            'sentence': self.sentence_corpus[index],
            'words': self.words[index].split('\t'),
            'terms':self.terms[index].split('\t'),
            'length':self.lengths[index].split('\t')
        }

        return sample

def collate_fn(batch: List[Dict[str, any]],tokenizer=BertTokenizerFast.from_pretrained(r'/home/sirui/WMM/Car/model/Encoder/BERT_BASE')) -> Dict[str, torch.Tensor]:
    sentences = [example["sentence"] for example in batch]
    # 将句子转换为tokens
    tokens = tokenizer(sentences, truncation=True, padding=True, return_tensors='pt', max_length=512)
    # class_infor=[example["class_index"].strip().split('\t') for example in batch]
    terms = [example['terms'] for example in batch]
    length = [example['length'] for example in batch]
    words = [example['words'] for example in batch]
    tokens['sentence'] = sentences
    tokens['labels'] = tokens['input_ids'].clone()
    # tokens['cls_label']=class_infor
    tokens['term'] = terms
    tokens['length'] = length
    tokens['words'] = words
    probability_matrix = torch.full(tokens['labels'].shape, 0.15)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in tokens['labels'].tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    Mask = []
    mask_index = []
    tokens['labels'] = tokens['labels'].tolist()
    for j in range(len(tokens['labels'])):
        val = tokens['labels'][j]
        term = list(''.join(terms[j]))
        word =list(''.join(words[j]))
        con = []
        index = []
        idx = 0
        word_idx = 0
        for i in range(len(val)):
            v = val[i]
            if v == 491 or v == 9177:
                tokens['labels'][j][i] = tokenizer.convert_tokens_to_ids(term[idx])
                idx = idx + 1
                con.append(True)
            else:
                if v == 482 or v ==13620:
                    if word_idx >= len(word):
                        #print(tokens['labels'][j])
                        print(tokens['sentence'][j])
                        #print(tokens['input_ids'][j])
                        print(word)
                        print(len(word))
                    tokens['labels'][j][i] = tokenizer.convert_tokens_to_ids(word[word_idx])
                    word_idx = word_idx + 1
                    con.append(True)
                else:
                    con.append(False)
        mask_index.append(index)
        Mask.append(con)
    tokens['labels'] = torch.tensor(tokens['labels'])
    Mask = torch.tensor(Mask)
    masked_indices = Mask
    tokens['masked_index'] = mask_index
    # masked_indices = torch.bernoulli(probability_matrix).bool()#伯努利采样
    tokens['labels'][~masked_indices] = -100  # We only compute loss on masked tokens
    indices_replaced = torch.bernoulli(torch.full(tokens['labels'].shape, 1.0)).bool() & masked_indices
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    # indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    tokens['input_ids'][indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    return tokens
