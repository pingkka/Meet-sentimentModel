import test_class
# import test_class_ori
# import test_audio_class
# import test_text_class
import numpy as np
import os
import csv
import re
from collections import Counter
import time


#test_file.csv
#filename,text,emotion,gender
#5e258fd1305bcf3ad153a6a4,어 청소 니가 대신 해 줘,0,male

labels = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]
classification = test_class.LanoiceClassification()
# classification = test_audio_class.audioClassification()
# classification = test_text_class.textClassification()
directories = os.listdir("test_wav")
print(directories)

f = open("csv/test_file.csv", 'r', encoding='utf-8-sig')
rdr = csv.reader(f)

correct = 0
total = 0
sentiment_total = [0] * len(labels)
sentiment_correct = [0] * len(labels)
sentiment_gender_correct = [0] * 2
gender_correct = 0

time_arr = []
for line in rdr:
    if (line[0] + ".wav") not in directories: continue
    #if (line[3] == "male"): continue
    audio_path = "test_wav/" + line[0]+".wav"
    text = line[1]
    start_time = time.time()
    # result = classification.textClassification2(text)
    result, gender_result = classification.classify(audio_path, text)
    # result = classification.classify(audio_path,text)
    # result, gender_result = classification.classify(audio_path)
    end_time = time.time()
    sentiment_total[labels.index(result)]+=1
    print(total)
    print("감정, 성별 : " + labels[int(line[2])] + ", " + line[3])
    print("예측 결과 : " + result + ", " + gender_result) #감정 분석 결과
    # print("예측 결과 : " + result) #감정 분석 결과
    print("실행 속도 : {} sec".format((end_time - start_time)))
    time_arr.append(end_time - start_time)

    if (result == labels[int(line[2])]):
        correct += 1
        sentiment_correct[int(line[2])] +=1
        if (line[3] == "male"):
            sentiment_gender_correct[0] +=1
        else :
            sentiment_gender_correct[1] +=1
    if (gender_result == line[3].lower()):
        gender_correct +=1


    total+=1



print("-------------------------------------------")
print("분류 개수 : ", sentiment_total)
print("Accuracy : " + str(correct/total*100) + "%")
print("-------------감정별 정확도----------------")
for i in range(0, len(sentiment_correct)):
    print(labels[i] + " : " + str(sentiment_correct[i]/(total/len(labels)) * 100) + "%")
print("-------------성별 감정 정확도-------------------")
print("Male" + " : " + str(sentiment_gender_correct[0]/(total/len(sentiment_gender_correct)) * 100) + "%")
print("Female" + " : " + str(sentiment_gender_correct[1]/(total/len(sentiment_gender_correct)) * 100) + "%")

print("-------------성별 분류 정확도-------------------")
print("Accuracy : " + str(gender_correct/total * 100) + "%")

print("---------------------------------------------")
print("가장 빠른 실행 속도 : " + str(min(time_arr)) + " sec")
print("가장 느린 실행 속도 : " + str(max(time_arr)) + " sec")
print("평균 실행 속도 : " + str(sum(time_arr)/len(time_arr)) + " sec")



