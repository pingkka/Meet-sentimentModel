########################################################
#음성 모델 훈련
########################################################


import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pickle
from xgboost import XGBClassifier
import csv
from collections import Counter


mslen = 22050

data = []

max_fs = 0
labels = []

emotions = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]
label_max = 1900
label = 0

path_data = ["4_wav", "5_wav"]
file_name = ['audio_sentiment.csv', 'audio_sentiment5.csv']

#file_name = ['audio_sentiment_test.csv']


def most_common_top_1(candidates):
    #배열에서 가장 많이 나온 값 출력 (동점일 시 더 앞에 있는 index로 출력함)
    assert isinstance(candidates, list), 'Must be a list type'
    if len(candidates) == 0: return None
    return Counter(candidates).most_common(n=1)[0][0]


#ai hub에서 받아온 데이터가 담긴 디렉토리의 음성 파일들을 가져와 특징 데이터 추출 (저작권 문제로 데이터셋은 github에 업로드 하지 않음)
i = 0
feature_all = np.array([])
for a in range(0, len(file_name)) :
    directories = os.listdir(path_data[a])

    print(directories)
    f = open(file_name[a],'r',encoding='utf-8-sig')
    rdr = csv.reader(f)

    #음성 데이터셋의 감정을 분류한 csv 파일을 읽어 가장 많이 투표된 감정으로 라벨링 (5명이 각자 음성에 맞는 데이터를 기록한 파일임)
    for line in rdr:
        if (line[0] + ".wav") not in directories: continue
        print(line[0])
        file_path = path_data[a]+"/" + line[0] + ".wav"

        X, sr = librosa.load(file_path, sr=None)

        sentiment_line = []
        sentiment_line.append(line[3])
        sentiment_line.append(line[5])
        sentiment_line.append(line[7])
        sentiment_line.append(line[9])
        sentiment_line.append(line[11])
        most_sentiment = most_common_top_1(sentiment_line)
        print(most_sentiment)
        if most_sentiment == "Neutral" :  label = 0
        elif most_sentiment == "Happiness" : label = 1
        elif most_sentiment == "Angry" : label = 2
        elif most_sentiment == "Sadness" : label = 3
        elif most_sentiment == "Disgust" : label = 4
        elif most_sentiment == "Surprise" : label = 5
        else : label = 6

        if (labels.count(label) > label_max) : continue
        else : labels.append(label)




        stft = np.abs(librosa.stft(X))

        ############# EXTRACTING AUDIO FEATURES (음성 특징 데이터 추출) #############
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40),axis=1)

        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)

        mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T,axis=0)

        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr,fmin=0.5*sr* 2**(-6)).T,axis=0)

        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sr*2).T,axis=0)

        if (i == 0) : feature_all = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        else :
            features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            feature_all = np.vstack([feature_all, features])

        i+=1

    f.close()



#추가로 수집한 joy
directories = os.listdir("joy_wav")
print(directories)

for a in directories:
    labels.append(1)
    file_path = "joy_wav/" + a

    X, sr = librosa.load(file_path, sr=None)
    stft = np.abs(librosa.stft(X))

    ############# EXTRACTING AUDIO FEATURES #############
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40), axis=1)

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

    mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T, axis=0)

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr, fmin=0.5 * sr * 2 ** (-6)).T, axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr * 2).T, axis=0)

    if (i == 0):
        feature_all = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    else:
        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        feature_all = np.vstack([feature_all, features])



#추가로 수집한 surprise
directories = os.listdir("sur_wav")
print(directories)

for a in directories:
    labels.append(5)
    file_path = "sur_wav/" + a

    X, sr = librosa.load(file_path, sr=None)
    stft = np.abs(librosa.stft(X))

    ############# EXTRACTING AUDIO FEATURES #############
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40), axis=1)

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

    mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T, axis=0)

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr, fmin=0.5 * sr * 2 ** (-6)).T, axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr * 2).T, axis=0)

    if (i == 0):
        feature_all = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    else:
        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        feature_all = np.vstack([feature_all, features])


#감정 별 데이터 개수 출력
for i in range(0, len(emotions)) :
    print(emotions[i] + " : " + str(labels.count(i)))



#라벨 값 저장
from copy import deepcopy
y = deepcopy(labels)
for i in range(len(y)):
    y[i] = int(y[i])


#X_train,X_test,y_train,y_test = train_test_split(feature_all,one_hot_encode,test_size = 0.3,shuffle=True, random_state=20)

#데이터셋 split
X_train2,X_test2,y_train2,y_test2 = train_test_split(feature_all,y,test_size = 0.3,shuffle=True, random_state=30)
eval_s = [(X_train2, y_train2),(X_test2,y_test2)]


########################### MODEL  ###########################

model3 = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4)
model3.fit(X_train2,y_train2, eval_set = eval_s)
model3.evals_result()
score = cross_val_score(model3, X_train2, y_train2, cv=5)
y_pred3 = model3.predict(X_test2)


count = 0
for i in range(y_pred3.shape[0]):
    if y_pred3[i] == y_test2[i]:
        count+=1

print('Accuracy for model 3 : ' + str((count / y_pred3.shape[0]) * 100))




