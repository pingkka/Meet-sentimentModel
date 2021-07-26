########################################################
#텍스트/음성 파일로 예측하는 코드 (test_**.py import 후 사용)
########################################################

import test_class
import time
#import test_text_class
#import test_audio_class

#5e2ac3d55807b852d9e01fd6,우리 아빠 어제 술먹고 또 사고쳤어.,anger,Angry,2,Angry,2,Angry,2,Angry,1,Angry,1,33,female
audio = "4_wav/5e2ac3d55807b852d9e01fd6.wav" #음성 파일
text = "우리 아빠 어제 술먹고 또 사고쳤어" #텍스트 파일
classification = test_class.LanoiceClassification()
start_time = time.time()
result = classification.classify(audio, text)
end_time = time.time()
print(result) #감정 분석 결과
print("실행 속도 : {} sec".format((end_time-start_time)))

#5e2ac6f85807b852d9e01ffe,고등학교 때 엄청 친한 사이였어. 근데 이럴 줄은 몰랐다.,anger,Neutral,0,Neutral,0,Neutral,0,Neutral,0,Neutral,0,37,male
audio = "4_wav/5e2ac6f85807b852d9e01ffe.wav" #음성 파일
text = "고등학교 때 엄청 친한 사이였어. 근데 이럴 줄은 몰랐다" #텍스트 파일
start_time = time.time()
result = classification.classify(audio, text)
end_time = time.time()
print(result) #감정 분석 결과
print("실행 속도 : {} sec".format((end_time-start_time)))
