import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from tqdm.notebook import tqdm
import model
import matplotlib.pyplot as plt
import matplotlib



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
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        return input_ids, attention_mask, y


train_dataset = HwangariDataset("traindata.csv")
test_dataset = HwangariDataset("testdata.csv")






# GPU 사용
device = torch.device("cuda")


model = model.HwangariSentimentModel.from_pretrained("monologg/koelectra-base-v3-discriminator").to(device)

epochs = 1
batch_size = 64


# 모델 레이어 보기
model

optimizer = AdamW(model.parameters(), lr=1e-5)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

losses = []
accuracies = []

for i in range(epochs):
    total_loss = 0.0
    correct = 0
    total = 0
    batches = 0

    model.train()

    for input_ids_batch, attention_masks_batch, y_batch in tqdm(train_loader):
        optimizer.zero_grad()
        y_batch = y_batch.to(device)
        y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
        loss = F.cross_entropy(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(y_pred, 1)
        correct += (predicted == y_batch).sum()
        total += len(y_batch)

        batches += 1
        if batches % 100 == 0:
            print("Batch Loss:", total_loss, "Accuracy:", correct.float() / total)

    losses.append(total_loss)
    accuracies.append(correct.float() / total)
    print("Train Loss:", total_loss, "Accuracy:", correct.float() / total)



losses, accuracies

model.eval()

test_correct = 0
test_total = 0





for input_ids_batch, attention_masks_batch, y_batch in tqdm(test_loader):
  y_batch = y_batch.to(device)
  y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
  _, predicted = torch.max(y_pred, 1)
  test_correct += (predicted == y_batch).sum()
  test_total += len(y_batch)



print("Accuracy:", test_correct.float() / test_total)


#모델 저장하기
torch.save(model.state_dict(), "real_model.pt")


