""" Multi span QA model"""
from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch import nn
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.bert.configuration_bert import BertConfig


@dataclass
class MultiSpanQuestionAnsweringModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    predicted_relations: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def margin_loss(inputs, threshold=0.1, margin=0.1):
    return torch.maximum(torch.zeros_like(inputs), threshold - inputs + margin)


# TODO: pos_margin_loss / neg_margin_loss
# threshold: 0.5 / -0.5


def append_1_length_zeros(tensor):
    B, _, D = tensor.shape
    zero_embed = tensor.new_zeros((B, 1, D))
    fixed_tensor = torch.cat((tensor, zero_embed), 1)
    return fixed_tensor


class BertForMultiSpanQuestionAnswering(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        print("******** config.num_labels:", config.num_labels)
        self.num_labels = config.num_labels  # 3, 1: start, 2: end, 0: neither
        self.rel_num_labels = 6  # 0: (0, 0), 1: (0, 1), 2: (1, 0), 3:(1, 1), 4: (2, 0)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.prev_cur_rel_outputs = nn.Linear(2 * config.hidden_size, self.rel_num_labels)
        self.relations = nn.ModuleList()
        # (O, O) -> 3
        # (O, B) -> 4
        # (B, I) (I, I) -> 8 12
        # (B, B) (I, B) -> 6 10
        # (B, O) (I, O) -> 5 9
        self.relations.append(nn.Linear(2 * config.hidden_size, config.hidden_size))  # relation (0, 0) -> 0
        self.relations.append(nn.Linear(2 * config.hidden_size, config.hidden_size))  # relation (0, 1) -> 1
        self.relations.append(nn.Linear(2 * config.hidden_size, config.hidden_size))  # relation (1, 0) -> 2
        self.relations.append(nn.Linear(2 * config.hidden_size, config.hidden_size))  # relation (1, 1) -> 3
        self.relations.append(nn.Linear(2 * config.hidden_size, config.hidden_size))
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            positive_start_positions: Optional[torch.Tensor] = None,
            positive_end_positions: Optional[torch.Tensor] = None,
            seo_targets: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultiSpanQuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        total_loss = None
        # last layer hidden states of the previous token: (batch_size, seq_len, hidden_size)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)  # (batch_size, seq_len, 2)

        if positive_start_positions is not None and positive_end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = logits.size(1)
            positive_start_positions = positive_start_positions.clamp(0, ignored_index)
            positive_end_positions = positive_end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(
                ignore_index=ignored_index,
                weight=torch.tensor([1.0, 1.0, 1.0]).to(logits.device),
                reduction='none'
            )
            # loss1:
            # (batch_size, ) -> (1,)
            class_loss = loss_fct(logits.reshape(-1, self.num_labels), seo_targets.reshape(-1)).mean()

            if total_loss is None:
                total_loss = class_loss
            else:
                total_loss += class_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MultiSpanQuestionAnsweringModelOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
