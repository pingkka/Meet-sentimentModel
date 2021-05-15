import pandas as pd
import torch
from transformers import AutoTokenizer, ElectraForSequenceClassification
import model
import re
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout, Flatten, Embedding
import pickle
from xgboost import XGBClassifier
import csv
from collections import Counter
from keras.models import model_from_json, load_model


emotions = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]
'''
json_file = open("mlp_model_tanh_adadelta.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
'''
# 파일명
filename = 'xgb_model.model'

# 모델 불러오기
loaded_model = pickle.load(open(filename, 'rb'))




########################### TESTING ###########################
test_file_path = "AnyConv.com__a1.wav"
X,sr = librosa.load(test_file_path, sr = None)
stft = np.abs(librosa.stft(X))

############# EXTRACTING AUDIO FEATURES #############
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40),axis=1)

chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)

mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T,axis=0)

contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr,fmin=0.5*sr* 2**(-6)).T,axis=0)

tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sr*2).T,axis=0)

features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])

#feature_all = np.array([])

#feature_all = np.vstack([feature_all,features])




x_chunk = np.array(features)
x_chunk = x_chunk.reshape(1,np.shape(x_chunk)[0])
y_chunk_model1 = loaded_model.predict(x_chunk)
y_chunk_model1_proba = loaded_model.predict_proba(x_chunk)
index = np.argmax(y_chunk_model1)

print("-----<Accuracy>------")
for proba in range(0, len(y_chunk_model1_proba[0])):
    print(emotions[proba]+  " : " + str(y_chunk_model1_proba[0][proba]))

print('\nEmotion:',emotions[int(y_chunk_model1[0])])

labels = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]
none_words = ["안싫", "안 싫", "안무서", "안놀람", "안놀랐", "안행복", "안기뻐", "안빡","안우울", "안짜증", "안깜짝", "안무섭"]
pass_words = ["안좋", "안 좋"]
senti_loss = [5.0, 4.0, 6.5, 6.5, 9.0, 8.0]

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
text = "빨리 졸작 끝났으면 좋겠다"
enc = tokenizer.encode_plus(text)
inputs = tokenizer(
  text,
  return_tensors='pt',
  truncation=True,
  max_length=256,
  pad_to_max_length=True,
  add_special_tokens=True
)

print(tokenizer.tokenize(tokenizer.decode(enc["input_ids"])))



# GPU 사용
device = torch.device("cuda")
model = model.HwangariSentimentModel.from_pretrained("Kyuyoung11/haremotions-v1").to(device)


model.eval()

input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)
output = model(input_ids.to(device), attention_mask.to(device))[0]
_, prediction = torch.max(output, 1)
print(output)




label_loss_str = str(output).split(",")

label_loss = [float(x.strip().replace(']','')) for x in label_loss_str[1:7]]
print(label_loss)
print(sum(label_loss))





print(f'Review text : {text}')

pre_result = int(re.findall("\d+",str(prediction))[0])
#손실함수 값이 4.0이상인게 없으면 무감정(none)으로 분류
result = 0
if label_loss[pre_result-1] >= senti_loss[pre_result-1]:
  result = pre_result


#안이 들어간 말로 결과가 나왔을 경우 가장 큰 값을 무시함 or 아예 무감정으로 분류되도록 함
for i in none_words:
  if i in text:
    result = 0
for j in pass_words:
  if j in text:
    label_loss[pre_result - 1] = 0
    result = label_loss.index(max(label_loss)) + 1


print(f'Sentiment : {labels[result]}')

print("\n<감정 별 손실 함수 값>")
for i in range(0,6):
  print(labels[i+1], ":", label_loss[i])