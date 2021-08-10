
########################################################
#오디오 파일로만 예측하는 코드
########################################################

import numpy as np
import librosa
import pickle
import preprocessing_audio



class audioClassification():
    def __init__(self):
        self.emotions = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]

        # 파일명
        self.filename = 'audio_gender_model/xgb_1_300_0.1_6.model'

        # 모델 불러오기
        self.loaded_model = pickle.load(open(self.filename, 'rb'))



    def classify(self, file):

        mfccs, chroma, mel, contrast, tonnetz = preprocessing_audio.extract_mfcc(file)

        features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])





        x_chunk = np.array(features)
        x_chunk = x_chunk.reshape(1,np.shape(x_chunk)[0])
        y_chunk_model1 = self.loaded_model.predict(x_chunk)
        y_chunk_model1_proba = self.loaded_model.predict_proba(x_chunk)
        index = np.argmax(y_chunk_model1)

        print(y_chunk_model1_proba)
        print(y_chunk_model1)
        print(index)

        #print("-----<Accuracy>------")
        #for proba in range(0, len(y_chunk_model1_proba[0])):
        #    print(self.emotions[proba]+  " : " + str(y_chunk_model1_proba[0][proba]))

        #print('\nEmotion:',self.emotions[int(y_chunk_model1[0])])
        return str(self.emotions[int(y_chunk_model1[0])])


classification = audioClassification()
audio_path = "4_wav/5e37dc03ee8206179943cb41.wav"
result = classification.classify(audio_path)
print(result)