import pandas as pd
import torch
from transformers import AutoTokenizer, ElectraForSequenceClassification
import model
import re


labels = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
text = "무서워"
inputs = tokenizer(
  text,
  return_tensors='pt',
  truncation=True,
  max_length=256,
  pad_to_max_length=True,
  add_special_tokens=True
)


# GPU 사용
device = torch.device("cuda")
model = model.HwangariSentimentModel.from_pretrained("Kyuyoung11/haremotions-v1").to(device)

#model.load_state_dict(torch.load("real_model.pt"))

model.eval()

input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)
output = model(input_ids.to(device), attention_mask.to(device))[0]
_, prediction = torch.max(output, 1)



label_loss_str = str(output).split(",")
label_loss = [float(x.strip().replace(']','')) for x in label_loss_str[1:7]]




print(f'Review text : {text}')

#손실함수 값이 4.0이상인게 없으면 무감정(none)으로 분류
nlabel = 0
for i in label_loss:
  if i > 4.0 :
    nlabel = int(re.findall("\d+",str(prediction))[0])
    break



print(f'Sentiment : {labels[nlabel]}')

print("\n<감정 별 손실 함수 값>")
for i in range(0,6):
  print(labels[i+1], ":", label_loss[i])