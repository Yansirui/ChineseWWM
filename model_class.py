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

def read_json(json_path):
    with open(json_path,'r',encoding='utf-8') as f:
        str=f.read()
        data=json.loads(str)
        return data
        
class MyOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    terminate_states: Optional[Tuple[torch.FloatTensor]] = None
#初始化模型的参数文件存放地址
check_point=r'/home/sirui/WMM/Car/model/Encoder/BERT-wwm-ext'
tokenizer=AutoTokenizer.from_pretrained(check_point)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Random_Whole_model(nn.Module):
    def __init__(self):
        super(Random_Whole_model, self).__init__()
        bert_model = BertForMaskedLM.from_pretrained(check_point)
        self.bert = bert_model

    def forward(self, input_ids, attention_mask, token_type_ids,labels=None):
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        mlm_input = outputs.last_hidden_state
        mlm_logits = self.bert.cls(mlm_input)
        loss_fct = CrossEntropyLoss()  # -100 index = padding token  -100为忽略标志
        masked_lm_loss = None
        if labels != None:
            masked_lm_loss = loss_fct(mlm_logits.view(-1, self.bert.config.vocab_size), labels.view(-1))
        loss = masked_lm_loss
        return MaskedLMOutput(
            loss=loss,
            logits=mlm_logits,
            hidden_states=mlm_input,
            attentions=None,
        )

    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = cls(**kwargs)
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
