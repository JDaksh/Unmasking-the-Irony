!pip uninstall -y transformers accelerate
!pip install transformers accelerate

!pip uninstall -y transformers
!pip install transformers==4.28.0

!pip install sentencepiece

!pip install imbalanced-learn

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import XLNetTokenizer, XLNetForSequenceClassification, TrainingArguments, Trainer

from google.colab import drive
drive.mount('/content/drive')

train_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Sarcasm/dataset/isarcasm/preprocessed/train.csv')
test_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Sarcasm/dataset/isarcasm/preprocessed/test.csv')

train_data['sarcastic'].value_counts()

test_data['sarcastic'].value_counts()

from imblearn.over_sampling import RandomOverSampler
import pandas as pd

# Separate the features and labels
X = train_data['tweet']  # Features (text)
y = train_data['sarcastic']  # Labels (0 or 1)

# Create an instance of RandomOverSampler
oversampler = RandomOverSampler()

# Perform random oversampling
X_resampled, y_resampled = oversampler.fit_resample(X.values.reshape(-1, 1), y)

# Convert the oversampled data back to DataFrame format
df_resampled = pd.DataFrame({'tweet': X_resampled.flatten(), 'sarcastic': y_resampled})

# Now you have an oversampled dataset in df_resampled

df_resampled['sarcastic'].value_counts()

X_train = df_resampled['tweet']
y_train = df_resampled['sarcastic']
X_test = test_data['tweet']
y_test = test_data['sarcastic']

class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class SarcasmTestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item
    def __len__(self):
        return len(self.encodings)

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(labels, pred)

    return {"accuracy": accuracy,"f1_score":f1}

X_train = X_train.tolist()
X_test = X_test.tolist()
y_train = y_train.tolist()
y_test = y_test.tolist()

model_name = 'detecting-sarcasm'
task='sentiment'
MODEL = 'xlnet-base-cased'

tokenizer = XLNetTokenizer.from_pretrained(MODEL,num_labels=2, loss_function_params={"weight": [0.75, 0.25]})

train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors='pt')
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, return_tensors='pt')

train_dataset = SarcasmDataset(train_encodings, y_train)
test_dataset = SarcasmDataset(test_encodings, y_test)

training_args = TrainingArguments(
    output_dir='./res', num_train_epochs=5, per_device_train_batch_size=32, warmup_steps=500, weight_decay=0.01,logging_dir='./logs4'
)

model = XLNetForSequenceClassification.from_pretrained(MODEL)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    compute_metrics = compute_metrics,
)
trainer.train()

preds = trainer.predict(test_dataset)
preds = np.argmax(preds.predictions[:, 0:2], axis=-1)

from sklearn.metrics import classification_report
report = classification_report(y_test, preds, zero_division=1)
print(report)

from sklearn.metrics import confusion_matrix

# create confusion matrix
cm = confusion_matrix(y_test, preds)

# print confusion matrix
print("Confusion Matrix:")
print(cm)

import seaborn as sns
import matplotlib.pyplot as plt

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Example test texts
test_texts = ['the irony of be ask to write an immaginary sarcastic tweet with imaginary incorrectly spell']

# Function to predict sarcasm for individual texts
def predict_sarcasm(text):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move tensors to the same device as the model
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_labels = np.argmax(logits.detach().cpu().numpy(), axis=1)  # Move logits to CPU for numpy operations
    return predicted_labels[0]

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model.to(device)

# Move the test dataset tensors to the same device as the model
test_dataset = SarcasmDataset({k: v.to(device) for k, v in test_dataset.encodings.items()}, test_dataset.labels)

# Perform prediction on test texts
for text in test_texts:
    predicted_label = predict_sarcasm(text)
    sarcasm_label = "Sarcastic" if predicted_label == 1 else "Not sarcastic"
    print(f"Text: {text}")
    print(f"Predicted Label: {sarcasm_label}")
    print()