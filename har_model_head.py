from transformers.activations import get_activation
import torch.nn as nn

class ElectraClassificationHead(nn.Module):
  """Head for sentence-level classification tasks."""

  def __init__(self, config, num_labels):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, 4 * config.hidden_size)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.out_proj = nn.Linear(4 * config.hidden_size, num_labels)


  def forward(self, features, **kwargs):
    x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
    x = self.dropout(x)
    x = self.dense(x)
    x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
    x = self.dropout(x)
    x = self.out_proj(x)
    return x

  #
  # def __init__(self, config, num_labels):
  #   super().__init__()
  #   self.dropout = nn.Dropout(config.hidden_dropout_prob)
  #   self.out_proj = nn.Linear(config.hidden_size, num_labels)
  #
  # def forward(self, features, **kwargs):
  #   x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
  #   x = self.dropout(x)
  #   x = self.out_proj(x)
    return x