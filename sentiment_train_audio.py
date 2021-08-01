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
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Embedding
import preprocessing_audio



def train():

    feature_all, one_hot_encode, y= preprocessing_audio.data_preprocessing_total()

    X_train, X_test, y_train, y_test = train_test_split(feature_all, one_hot_encode, test_size=0.3, shuffle=True,
                                                        random_state=20)

    ########################### MODEL 1 ###########################
    model = Sequential()
    model.add(Dense(X_train.shape[1],input_dim =X_train.shape[1], activation ='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(y_train.shape[1],activation ='softmax'))
    model.compile(optimizer = "Adam", loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    # Since the dataset already takes care of batching,
    # we don't pass a `batch_size` argument.
    model.fit(X_train,y_train, epochs=500, batch_size = 16,verbose=1)
    model.evaluate(X_test,y_test)

    #Hugging face에 업로드할 파일 저장
    model.save('audio_model/audio_50016_f.h5')

    y_pred_model1 = model.predict(X_test)
    y2 = np.argmax(y_pred_model1,axis=1)
    y_test2 = np.argmax(y_test , axis = 1)

    count = 0
    for i in range(y2.shape[0]):
        if y2[i] == y_test2[i]:
            count+=1

    print('Accuracy for model 1 : ' + str((count / y2.shape[0]) * 100))



    #데이터셋 split
    X_train2,X_test2,y_train2,y_test2 = train_test_split(feature_all,y,test_size = 0.3,shuffle=True, random_state=30)
    eval_s = [(X_train2, y_train2),(X_test2,y_test2)]


    ########################### MODEL 2###########################

    model3 = XGBClassifier(n_estimators=500, learning_rate=0.2, max_depth=4)
    model3.fit(X_train2,y_train2, eval_set = eval_s)
    model3.evals_result()
    score = cross_val_score(model3, X_train2, y_train2, cv=5)
    y_pred3 = model3.predict(X_test2)


    count = 0
    for i in range(y_pred3.shape[0]):
        if y_pred3[i] == y_test2[i]:
            count+=1

    print('Accuracy for model 3 : ' + str((count / y_pred3.shape[0]) * 100))


    # 파일명
    filename = 'audio_model/xgb_50024_f.model'

    # 모델 저장
    pickle.dump(model3, open(filename, 'wb'))


    ###################################################################################################
    ###################################################################################################

train()



