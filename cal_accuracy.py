import test_class
import numpy as np
import os
import csv
import re
from collections import Counter


#test_file.csv
#filename,text,emotion,gender
#5e258fd1305bcf3ad153a6a4,어 청소 니가 대신 해 줘,0,male

labels = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]
classification = test_class.LanoiceClassification()
directories = os.listdir("test_wav")
print(directories)

f = open("csv/test_file.csv", 'r', encoding='utf-8-sig')
rdr = csv.reader(f)

correct = 0
total = 0
for line in rdr:
    if (line[0] + ".wav") not in directories: continue
    audio_path = "test_wav/" + line[0]+".wav"
    text = line[1]
    result = classification.classify(audio_path, text)
    print(result) #감정 분석 결과
    if (result == labels[int(line[2])]):
        correct += 1
    total+=1

print("Accuracy : " + str(correct/total*100) + "%")




# audio = "4_wav/5e2ac3d55807b852d9e01fd6.wav" #음성 파일
# text = "우리 아빠 어제 술먹고 또 사고쳤어" #텍스트 파일
# classification = test_class.LanoiceClassification()
# result = classification.classify(audio, text)
#print(result) #감정 분석 결과