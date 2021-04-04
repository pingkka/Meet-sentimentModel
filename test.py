import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.activations import get_activation
from transformers import (
  ElectraPreTrainedModel,
  ElectraModel,
  ElectraConfig,
  ElectraTokenizer
)
import model

model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

text = "재밌다! 내용도 신선하고 의미도있으며 연기도 좋고 영상도좋다 즐거운 시간이었다"
text2 = "다음에는 무슨 영화 보지?"
encoded_dict = tokenizer(text, text2)
enc = tokenizer.encode_plus(text)
print(enc.keys())
print(encoded_dict['token_type_ids'])

# sequence_output, pooled_output, (hidden_states), (attentions)
out = model(torch.tensor(enc["input_ids"]).unsqueeze(0), torch.tensor(enc["attention_mask"]).unsqueeze(0))
print(out[0].shape)  # torch.size(batch size, 토큰화된 텍스트 길이, model output hidden size
# torch.Size([1, 22, 768])

token_representations = model(torch.tensor(enc["input_ids"]).unsqueeze(0))[0][0]
print(enc["input_ids"])
print(enc["attention_mask"])
print(enc['token_type_ids'])
print(tokenizer.decode(enc["input_ids"]))
print(tokenizer.tokenize(tokenizer.decode(enc["input_ids"])))
print(f"Length: {len(enc['input_ids'])}")
print(token_representations.shape)






classifier = model.HwangariSentimentModel(ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator"),5)

X = torch.tensor(enc["input_ids"]).unsqueeze(0)
attn = torch.tensor(enc["attention_mask"]).unsqueeze(0)
toktype = torch.tensor(enc["token_type_ids"]).unsqueeze(0)

print(classifier(X, attn, toktype))
