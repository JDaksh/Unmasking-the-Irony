"""## Import packages"""

!pip install transformers

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import drive
from torch import nn
from torch.optim import Adam
from transformers import GPT2Model, GPT2Tokenizer
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay

drive.mount('/content/drive/')

train_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Sarcasm/dataset/isarcasm/clean/train.csv')
test_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Sarcasm/dataset/isarcasm/clean/test.csv')

"""## Preprocessing data (text tokenization)"""

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

"""## Dataset class"""

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
labels = {0: 0, 1: 1}

class trainDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in train_df['sarcastic']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=128,
                                truncation=True,
                                return_tensors="pt") for text in train_df['tweet']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Get a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Get a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

"""## Split training dataset"""

from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

"""## Model building"""

class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, gpt_model_name:str):
        super(SimpleGPT2SequenceClassifier,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        self.fc1 = nn.Linear(hidden_size*max_seq_len, num_classes)


    def forward(self, input_id, mask):
        """
        Args:
                input_id: encoded inputs ids of sent.
        """
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size,-1))
        return linear_output

"""## Training"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

def train(model, train_data, val_data, learning_rate, epochs):
    train_dataset = trainDataset(train_data)
    val_dataset = trainDataset(val_data)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        all_predictions_train = []
        all_labels_train = []

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)

            model.zero_grad()

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            # Collect predictions and labels for F1 score calculation
            all_predictions_train.extend(output.argmax(dim=1).cpu().tolist())
            all_labels_train.extend(train_label.cpu().tolist())

            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0
        all_predictions_val = []
        all_labels_val = []

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

                # Collect predictions and labels for F1 score calculation
                all_predictions_val.extend(output.argmax(dim=1).cpu().tolist())
                all_labels_val.extend(val_label.cpu().tolist())

        train_f1 = f1_score(all_labels_train, all_predictions_train, average='weighted')
        val_f1 = f1_score(all_labels_val, all_predictions_val, average='weighted')

        print(f"Epoch: {epoch_num + 1} \
              | Train Loss: {total_loss_train / len(train_data):.3f} \
              | Train Accuracy: {total_acc_train / len(train_data):.3f} \
              | Train F1 Score: {train_f1:.3f} \
              | Val Loss: {total_loss_val / len(val_data):.3f} \
              | Val Accuracy: {total_acc_val / len(val_data):.3f} \
              | Val F1 Score: {val_f1:.3f}")

EPOCHS = 5

model = SimpleGPT2SequenceClassifier(hidden_size=768, num_classes=5, max_seq_len=128, gpt_model_name="gpt2")
LR = 1e-5

train(model, train_df, val_df, LR, EPOCHS)

"""## Evaluation"""

class testDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in test_df['sarcastic']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=128,
                                truncation=True,
                                return_tensors="pt") for text in test_df['tweet']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Get a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Get a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

def evaluate(model, test_data):
    test_dataset = testDataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    # Tracking variables
    predictions_labels = []
    true_labels = []

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

            # Add original labels
            true_labels += test_label.cpu().numpy().flatten().tolist()
            # Get predictions as a list
            predictions_labels += output.argmax(dim=1).cpu().numpy().flatten().tolist()

    test_f1 = f1_score(true_labels, predictions_labels, average='weighted')

    print(f'Test Accuracy: {total_acc_test / len(test_data):.3f}')
    print(f'Test F1 Score: {test_f1:.3f}')

    return true_labels, predictions_labels

true_labels, pred_labels = evaluate(model, test_df)

print(classification_report(true_labels, pred_labels))

# Plot confusion matrix.
fig, ax = plt.subplots(figsize=(8, 8))
cm = confusion_matrix(y_true=true_labels, y_pred=pred_labels, labels=range(len(labels)), normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels.keys()))
disp.plot(ax=ax)

