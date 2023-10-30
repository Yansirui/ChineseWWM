from typing import Optional, Tuple

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import BertModel, AutoTokenizer, BertForMaskedLM
import torch.nn as nn
import os
import json
import numpy as np
from transformers.modeling_outputs import MaskedLMOutput
import torch
from transformers.models.bert.modeling_bert import BertLMPredictionHead
from transformers.utils import ModelOutput
import torch.nn.functional as F


check_point=r'/home/sirui/WMM/Car/model/Encoder/macBERT'

#tokenizer=AutoTokenizer.from_pretrained(check_point)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class ClassificationModel(nn.Module):
    def __init__(self,num_labels):
        super(ClassificationModel,self).__init__()
        bert_model = BertForMaskedLM.from_pretrained(check_point).bert
        self.bert=bert_model
        self.num_labels=num_labels
        self.classifier=nn.Linear(768,self.num_labels)

    def forward(self,input_ids, attention_mask, token_type_ids,cls_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        cls_token = outputs[:, 0, :]
        logits = self.classifier(cls_token)
        if cls_labels==None:
            loss=None
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), cls_labels.view(-1))
        #补全

        return{
            'hidden_states':outputs,
            'logits':logits,
            'loss':loss
        }

    def from_pretrained(cls, pretrained_model_name_or_path, num_labels,**kwargs):
        model = cls(num_labels=num_labels,**kwargs)
        state_dict = torch.load(os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin'), map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        # config = {"num_labels": self.classifier.out_features, "bert_config": self.bert.config.to_dict()}
        config = {'bert': self.bert.config.to_dict()}
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
