import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from tqdm.notebook import tqdm
import model


label2int = {
    "joy" : 1,
    "annoy" : 2
}

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")

class HwangariDataset(Dataset):

    def __init__(self, csv_file):
        # 일부 값중에 NaN이 있음...
        self.dataset = pd.read_csv(csv_file, sep=',').dropna(axis=0)
        # 중복제거
        #self.dataset.drop_duplicates(subset=['document'], inplace=True)
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")

        print(self.dataset.describe())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, 0:2].values
        text = row[0] #text
        y = row[1] #label

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            pad_to_max_length=True,
            add_special_tokens=True
        )

        #print(text)
        #print(tokenizer.decode(inputs['input_ids'][0]))
        #print(tokenizer.tokenize(tokenizer.decode(inputs["input_ids"][0])))
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        return input_ids, attention_mask, y


train_dataset = HwangariDataset("regex_result_joy_annoy.csv")
print(train_dataset[0][0]) #input_ids
print(train_dataset[0][1]) #attention_mask