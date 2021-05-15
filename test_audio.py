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