########################################################
#텍스트만 예측하는 코드
########################################################

import torch
from transformers import AutoTokenizer
import har_model
import re


class textClassification():
    def __init__(self):
        self.labels = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
        # GPU 사용
        self.device = torch.device("cuda")
        self.model = har_model.HwangariSentimentModel.from_pretrained("Kyuyoung11/haremotions-v5").to(self.device)


    def textClassification(self, text):



        #enc = tokenizer.encode_plus(text)
        inputs = self.tokenizer(
          text,
          return_tensors='pt',
          truncation=True,
          max_length=256,
          pad_to_max_length=True,
          add_special_tokens=True
        )

        #print(tokenizer.tokenize(tokenizer.decode(enc["input_ids"])))






        self.model.eval()

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        output = self.model(input_ids.to(self.device), attention_mask.to(self.device))[0]
        _, prediction = torch.max(output, 1)

        label_loss_str = str(output).split(",")
        label_loss_str[0] = label_loss_str[0][9:]
        label_loss = [float(x.strip().replace(']', '')) for x in label_loss_str[0:7]]
        print("\n<Text Loss>")

        # pre_result = int(re.findall("\d+", str(prediction))[0])
        result = int(re.findall("\d+", str(prediction))[0])

        for i in range(0, len(label_loss)):
            print(self.labels[i], ":", label_loss[i])
        print(f'Sentiment : {self.labels[result]}')

        return str(self.labels[result])



# classification = textClassification()
# text = "난 그런거 너무 싫어"
# result = classification.textClassification(text)
# print(result)