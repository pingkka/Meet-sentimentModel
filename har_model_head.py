from transformers.activations import get_activation
import torch.nn as nn

class ElectraClassificationHead(nn.Module):
  """Head for sentence-level classification tasks."""

  def __init__(self, config, num_labels):
    super().__init__()
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.out_proj = nn.Linear(config.hidden_size, num_labels)

  def forward(self, features, **kwargs):
    x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
    x = self.dropout(x)
    x = self.out_proj(x)
    return x