import pandas as pd
import torch
from transformers import AutoTokenizer, ElectraForSequenceClassification
import model
import re


labels = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
text = "우린 어제 한국으로 돌아왔어요."
#3081
enc = tokenizer.encode_plus(text)
inputs = tokenizer(
  text,
  return_tensors='pt',
  truncation=True,
  max_length=256,
  pad_to_max_length=True,
  add_special_tokens=True
)

print(inputs['input_ids'])
print(tokenizer.tokenize(tokenizer.decode(enc["input_ids"])))



# GPU 사용
device = torch.device("cuda")
model = model.HwangariSentimentModel.from_pretrained("Kyuyoung11/haremotions-v1").to(device)


model.eval()

input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)
output = model(input_ids.to(device), attention_mask.to(device))[0]
_, prediction = torch.max(output, 1)



label_loss_str = str(output).split(",")
label_loss = [float(x.strip().replace(']','')) for x in label_loss_str[1:7]]





print(f'Review text : {text}')

pre_result = int(re.findall("\d+",str(prediction))[0])
#손실함수 값이 4.0이상인게 없으면 무감정(none)으로 분류
result = 0
for i in label_loss:
  if i > 4.0 :
    result = pre_result
    break

'''
#tokenize결과 중 '안'의 input_id : 3081
#안이 들어간 말로 결과가 나왔을 경우 가장 큰 값을 무시함
n = 3081
if n in inputs["input_ids"] :
  print("있음")
  print(max(label_loss[-nlabel]))
'''


print(f'Sentiment : {labels[result]}')

print("\n<감정 별 손실 함수 값>")
for i in range(0,6):
  print(labels[i+1], ":", label_loss[i])