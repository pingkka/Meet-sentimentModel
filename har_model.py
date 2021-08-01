
########################################################
#기존 KOELECTRA pretrained model에 은닉층 추가 후 분류하는 코드 (7가지의 감정으로 분류)
#(기존 KODELECTRA 모델은 감정을 긍정/부정으로만 분류함)
########################################################

import torch
import torch.nn as nn
import har_model_head
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import (
  ElectraPreTrainedModel,
  ElectraModel
)


# GPU 사용
device = torch.device("cuda")



class HwangariSentimentModel(ElectraPreTrainedModel):
  def __init__(self, config):
    super().__init__(config)
    self.num_labels = 7 #7개로 분류
    self.electra = ElectraModel(config).to(device)
    self.classifier = har_model_head.ElectraClassificationHead(config, self.num_labels).to(device)
    self.dropout = nn.Dropout(config.hidden_dropout_prob).to(device)

    self.init_weights()
  def forward(
          self,
          input_ids=None,
          attention_mask=None,
          token_type_ids=None,
          position_ids=None,
          head_mask=None,
          inputs_embeds=None,
          labels=None,
          output_attentions=None,
          output_hidden_states=None,
  ):
    discriminator_hidden_states = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds,
                                               output_attentions,output_hidden_states)

    sequence_output = discriminator_hidden_states[0]
    pooled_output = self.dropout(sequence_output)
    logits = self.classifier(pooled_output)

    outputs = (logits,) + discriminator_hidden_states[1:]  # add hidden states and attention if they are here

    if labels is not None:
      if self.num_labels == 1:
        #  We are doing regressionk,
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))
      else:
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      outputs = (loss,) + outputs

    return outputs  # (loss), (logits), (hidden_states), (attentions)