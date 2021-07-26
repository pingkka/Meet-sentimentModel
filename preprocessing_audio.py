import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pickle
from xgboost import XGBClassifier
import csv
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Embedding


def most_common_top_1(candidates):
    # 배열에서 가장 많이 나온 값 출력 (동점일 시 더 앞에 있는 index로 출력함)
    assert isinstance(candidates, list), 'Must be a list type'
    if len(candidates) == 0: return None
    return Counter(candidates).most_common(n=1)[0][0]


def extract_mfcc(file_path):
    X, sr = librosa.load(file_path, sr=None)
    stft = np.abs(librosa.stft(X))

    ############# EXTRACTING AUDIO FEATURES (음성 특징 데이터 추출) #############
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40),axis=1)

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)

    mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T,axis=0)

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr,fmin=0.5*sr* 2**(-6)).T,axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sr*2).T,axis=0)

    return mfccs,chroma,mel,contrast,tonnetz


def data_preprocessing():
    labels = []

    emotions = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]
    label_max = 1901
    label = 0

    path_data = ["4_wav", "5_wav"]
    path_data_2 = ["6_man_wav", "6_woman_wav"]
    file_name = ['audio_sentiment.csv', 'audio_sentiment5.csv']
    file_name_2 = 'audio_sentiment6.csv'

    ##########################################################################################
    # 음성 데이터셋 -> 특징 데이터 추출
    ##########################################################################################




    # ai hub - KETI 감성 대화 (github 업로드 x)
    i = 0
    feature_all = np.array([])
    for a in range(0, len(file_name)):
        directories = os.listdir(path_data[a])

        print(directories)
        f = open(file_name[a], 'r', encoding='utf-8-sig')
        rdr = csv.reader(f)

        # 음성 데이터셋의 감정을 분류한 csv 파일을 읽어 가장 많이 투표된 감정으로 라벨링 (5명이 각자 음성에 맞는 데이터를 기록한 파일임)
        for line in rdr:
            if (line[0] + ".wav") not in directories: continue
            print(line[0])
            file_path = path_data[a] + "/" + line[0] + ".wav"

            sentiment_line = []
            sentiment_line.append(line[3])
            sentiment_line.append(line[5])
            sentiment_line.append(line[7])
            sentiment_line.append(line[9])
            sentiment_line.append(line[11])
            most_sentiment = most_common_top_1(sentiment_line)
            print(most_sentiment)
            if most_sentiment == "Neutral":
                label = 0
            elif most_sentiment == "Happiness":
                label = 1
            elif most_sentiment == "Angry":
                label = 2
            elif most_sentiment == "Sadness":
                label = 3
            elif most_sentiment == "Disgust":
                label = 4
            elif most_sentiment == "Surprise":
                label = 5
            else:
                label = 6

            if (labels.count(label) > label_max):
                continue
            else:
                labels.append(label)

            mfccs, chroma, mel, contrast, tonnetz = extract_mfcc(file_path)

            if (i == 0):
                feature_all = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            else:
                features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
                feature_all = np.vstack([feature_all, features])

            i += 1

        f.close()

    # 모두의 말뭉치 - 감성 대화 음성 (github 업로드 x)
    if (labels.count(1) <= label_max):
        # 추가로 수집한 joy
        directories = os.listdir("joy_wav")
        print(directories)

        for a in directories:
            labels.append(1)
            file_path = "joy_wav/" + a

            mfccs, chroma, mel, contrast, tonnetz = extract_mfcc(file_path)

            if (i == 0):
                feature_all = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            else:
                features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
                feature_all = np.vstack([feature_all, features])
            i += 1

    if (labels.count(5) <= label_max):
        # 추가로 수집한 surprise
        directories = os.listdir("sur_wav")
        print(directories)

        for a in directories:
            labels.append(5)
            file_path = "sur_wav/" + a

            X, sr = librosa.load(file_path, sr=None)
            stft = np.abs(librosa.stft(X))

            mfccs, chroma, mel, contrast, tonnetz = extract_mfcc(file_path)

            if (i == 0):
                feature_all = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            else:
                features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
                feature_all = np.vstack([feature_all, features])
            i += 1

    #############ai hub - 감성 대화 데이터셋 (github 업로드 x) ######################
    for a in range(0, len(path_data_2)):
        directories = os.listdir(path_data_2[a])

        print(directories)
        f = open(file_name_2, 'r', encoding='utf-8-sig')
        rdr = csv.reader(f)

        for line in rdr:
            if (line[0] + ".wav") not in directories: continue
            # print(line[0])
            file_path = path_data_2[a] + "/" + line[0] + ".wav"

            if line[5] == "무감정":
                label = 0
            elif line[5] == "기쁨":
                label = 1
            elif line[5] == "분노":
                label = 2
            elif line[5] == "슬픔":
                label = 3
            elif line[5] == "혐오":
                label = 4
            elif line[5] == "놀람":
                label = 5
            elif line[5] == "무서움":
                label = 6

            if (labels.count(label) > label_max):
                continue
            else:
                labels.append(label)

            mfccs, chroma, mel, contrast, tonnetz = extract_mfcc(file_path)

            if (i == 0):
                feature_all = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            else:
                features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
                feature_all = np.vstack([feature_all, features])
            i += 1

        f.close()

    # 감정 별 데이터 개수 출력
    for i in range(0, len(emotions)):
        print(emotions[i] + " : " + str(labels.count(i)))

    # 라벨 값 저장
    from copy import deepcopy
    y = deepcopy(labels)
    for i in range(len(y)):
        y[i] = int(y[i])

    ###################################################################################################
    ###################################################################################################

    ###################################################################################################
    # 음성 감정 분석 모델 학습
    ###################################################################################################

    n_labels = len(y)
    n_unique_labels = len(np.unique(y))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    f = np.arange(n_labels)
    for i in range(len(f)):
        one_hot_encode[f[i], y[i] - 1] = 1
    print(feature_all)
    print(one_hot_encode)

    return feature_all, one_hot_encode, y