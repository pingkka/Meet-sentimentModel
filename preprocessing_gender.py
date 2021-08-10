########################################################
#음성 모델 데이터셋 읽고 전처리
########################################################

import numpy as np
import os
import librosa
import csv
from collections import Counter




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

def data_preprocessing_total():
    labels = []

    emotions = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]
    label_max = 3400
    label = 0




    male_count = 0
    female_count = 0

    path_data = ["4_wav", "5_wav", "7_wav"]
    path_data_2 = ["6_man_wav", "6_woman_wav"]
    file_name = ['audio_sentiment.csv', 'audio_sentiment5.csv', 'audio_sentiment7.csv']
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

            file_path = path_data[a] + "/" + line[0] + ".wav"

            if line[14].lower() == "male":
                label = 0
                male_count += 1
            elif line[14].lower() == "female":
                label = 1
                female_count += 1
            else :
                continue

            if (labels.count(label) >= label_max):
                continue

            labels.append(label)
            print(i, line[14])

            mfccs, chroma, mel, contrast, tonnetz = extract_mfcc(file_path)

            if (i == 0):
                feature_all = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            else:
                features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
                feature_all = np.vstack([feature_all, features])

            i += 1

        f.close()

    '''
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
    '''
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

            if line[3].lower() == "male":
                label = 0
            elif line[3] == "female":
                label = 1
            else :
                continue

            if (labels.count(label) >= label_max):
                continue
            labels.append(label)
            print(i, line[14])

            mfccs, chroma, mel, contrast, tonnetz = extract_mfcc(file_path)

            if (i == 0):
                feature_all = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            else:
                features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
                feature_all = np.vstack([feature_all, features])
            i += 1

        f.close()



    print(male_count)
    print(female_count)

    # 라벨 값 저장
    from copy import deepcopy
    y = deepcopy(labels)
    for i in range(len(y)):
        y[i] = int(y[i])

    ###################################################################################################
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



