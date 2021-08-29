
########################################################
#오디오 파일로만 예측하는 코드
########################################################

import numpy as np
import librosa
import pickle
import preprocessing_audio



class audioClassification():
    def __init__(self):
        self.labels = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]
        self.gender = ["male", "female"]

        # XGBoost 모델 사용시
        # 음성 모델 파일명
        # self.filename = 'audio_model/xgb_4_300_0.1_6.model'
        # 음성 모델 불러오기
        # self.audio_model = pickle.load(open(self.filename, 'rb'))

        # #Keras Sequential 모델 사용시
        # self.audio_model = load_model('audio_model/audio_4_unit25_0.1.h5')

        # gender 모델 파일명
        self.gender_file = 'audio_gender_model/xgb_1_300_0.1_6.model'

        # gender 모델 불러오기
        self.gender_model = pickle.load(open(self.gender_file, 'rb'))



    def classify(self, audio_path):

        ########################### TESTING ###########################
        # test_file_path = "5_wav/5f05fb0bb140144dfcff0184.wav"
        X, sr = librosa.load(audio_path, sr=None)
        stft = np.abs(librosa.stft(X))

        ############# EXTRACTING AUDIO FEATURES #############
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40), axis=1)

        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

        mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T, axis=0)

        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr, fmin=0.5 * sr * 2 ** (-6)).T, axis=0)

        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr * 2).T, axis=0)

        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])

        x_chunk = np.array(features)
        x_chunk = x_chunk.reshape(1, np.shape(x_chunk)[0])
        y_chunk_gender_proba = self.gender_model.predict_proba(x_chunk)
        gender_index = np.argmax(y_chunk_gender_proba)
        if (gender_index == 0):
            filename = 'audio_model/xgb_5_300_0.1_6_m.model'
        elif (gender_index == 1):
            filename = 'audio_model/xgb_1_300_0.1_6_f.model'
        # 음성 모델 불러오기
        audio_model = pickle.load(open(filename, 'rb'))

        y_chunk_model1_proba = audio_model.predict_proba(x_chunk)
        # y_chunk_model1_proba = audio_model.predict(x_chunk)
        print(y_chunk_model1_proba)
        index = np.argmax(y_chunk_model1_proba)

        print("----------------------------")
        print("<Audio Accuracy>")
        for proba in range(0, len(y_chunk_model1_proba[0])):
            print(self.labels[proba] + " : " + str(y_chunk_model1_proba[0][proba]))

        print('\nEmotion:', self.labels[int(index)])

        #print("-----<Accuracy>------")
        #for proba in range(0, len(y_chunk_model1_proba[0])):
        #    print(self.emotions[proba]+  " : " + str(y_chunk_model1_proba[0][proba]))

        #print('\nEmotion:',self.emotions[int(y_chunk_model1[0])])
        return self.labels[int(index)], self.gender[gender_index]


# classification = audioClassification()
# audio_path = "4_wav/5e37dc03ee8206179943cb41.wav"
# result = classification.classify(audio_path)
# print(result)