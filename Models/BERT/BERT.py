!pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

!pip install transformers

!pip uninstall -y transformers accelerate
!pip install transformers accelerate

!pip uninstall -y transformers
!pip install transformers==4.28.0

!pip install datasets

!pip install evaluate

from google.colab import drive
from datasets import load_dataset, Features, Value, ClassLabel, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, DataCollatorWithPadding, Trainer
import torch
import pandas as pd
import numpy as np
import evaluate
from sklearn.metrics import classification_report

MODEL_CHECKPOINT = "bert-base-uncased"

drive.mount("/content/drive")

class_names = ["0", "1"]

def preprocess_dataset(filename):
    data_features = Features(
        {
            "tweet": Value('string'),
            "sarcastic": ClassLabel(names=class_names)
        }
    )
    new_dataset = load_dataset("csv", data_files=filename, features=data_features, split='train')
    new_dataset = new_dataset.map(lambda example: {"tweet": example["tweet"], "sarcastic": str(example["sarcastic"])})
    new_dataset = new_dataset.train_test_split(train_size=0.7)

    return new_dataset

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, return_tensors = 'pt')
    return tokenizer

def load_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')
    print(device)
    return device

id2label = {0: "0", 1: "1"}
label2id = {"0": 0, "1": 1}

def preprocess_function(examples):
    tokenizer = load_tokenizer()
    model_inputs = tokenizer(examples["tweet"], truncation=True, max_length = 128, padding ="max_length")
    return model_inputs

def tokenize_dataset(new_dataset):
    tokenized_dataset = new_dataset.map(preprocess_function, batched = True)
    tokenized_dataset = tokenized_dataset.remove_columns(["tweet"])
    tokenized_dataset = tokenized_dataset.rename_column("sarcastic","labels")
    return tokenized_dataset

def load_data_collator():
    data_collator = DataCollatorWithPadding(load_tokenizer(), return_tensors ='pt')
    return data_collator

def set_training_args():
    args = TrainingArguments(
        output_dir = "/content/drive/MyDrive/Colab Notebooks/Sarcasm/checkpoints/Bert1/",
        evaluation_strategy = "epoch",
        logging_strategy = "epoch",
        learning_rate = 2e-5,
        do_eval = True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay = 0.01,
        save_strategy = "epoch",
        load_best_model_at_end=True,
    )
    return args

def compute_metric(eval_preds):
    f1_metric = evaluate.load("f1")
    predictions,labels = eval_preds
    predictions = np.argmax(predictions,axis=-1)
    f1_score = f1_metric.compute(predictions=predictions, references=labels, average = "micro")
    return f1_score

def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels = 2, id2label=id2label, label2id=label2id)
    model = model.to(load_device())
    return model

def set_trainer(tokenized_dataset):
    trainer = Trainer(
        load_model(),
        set_training_args(),
        train_dataset = tokenized_dataset['train'],
        eval_dataset = tokenized_dataset['test'],
        # data_collator=load_data_collator(),
        tokenizer=load_tokenizer(),
        compute_metrics=compute_metric,
    )
    return trainer

new_dataset = preprocess_dataset("/content/drive/MyDrive/Colab Notebooks/Sarcasm/dataset/isarcasm/preprocessed/train.csv")
tokenized_data = tokenize_dataset(new_dataset)
print(tokenized_data['train'][:2])
trainer = set_trainer(tokenized_data)
trainer.train()
predictions = trainer.predict(tokenized_data['test'])
f1_metric = evaluate.load("f1")
preds = np.argmax(predictions.predictions,axis=-1)
results = f1_metric.compute(predictions=preds, references=predictions.label_ids, average = "micro")
result_report = classification_report(preds, predictions.label_ids, target_names = class_names )
print(result_report)

"""# **Inference Pipeline**

"""

TRAINED_MODEL_CHECKPOINT = "/content/drive/MyDrive/Colab Notebooks/Sarcasm/checkpoints/Bert1/checkpoint-380"

def load_trained_model():
    test_model = AutoModelForSequenceClassification.from_pretrained(TRAINED_MODEL_CHECKPOINT, local_files_only = True, num_labels=2, id2label=id2label, label2id=label2id)
    test_model = test_model.to(load_device())
    return test_model

def load_test_tokenizer():
    test_tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_CHECKPOINT, return_tensors = 'pt')
    return test_tokenizer

def preprocess_test_dataset(filename):
    data_features = Features(
        {
            "tweet": Value('string'),
            "sarcastic":ClassLabel(names=class_names)
        }
    )
    test_dataset = load_dataset("csv", data_files = filename, features = data_features, split = 'train')
    test_dataset = test_dataset.map(lambda example: {"tweet": example["tweet"], "sarcastic": str(example["sarcastic"])})
    print(test_dataset)

    return test_dataset

def preprocess_test_function(examples):
    test_tokenizer = load_test_tokenizer()
    model_test_inputs = test_tokenizer(examples["tweet"], truncation=True, max_length = 128, padding ="max_length")
    return model_test_inputs

def tokenize_test_dataset(test_dataset):
    tokenized_test_dataset = test_dataset.map(preprocess_test_function, batched = True)
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(["tweet"])
    tokenized_test_dataset = tokenized_test_dataset.rename_column("sarcastic","labels")

    return tokenized_test_dataset

def set_test_args():
    test_args = TrainingArguments(
        output_dir = "/content/drive/MyDrive/Colab Notebooks/Sarcasm/checkpoints/Bert1",

        do_train = False,
        do_predict = True,
        do_eval = False,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

    )
    return test_args

def set_test_trainer(tokenized_test_dataset):
    test_trainer = Trainer(
        load_trained_model(),
        set_test_args(),

        eval_dataset = tokenized_test_dataset,
        # data_collator=load_data_collator(),
        tokenizer=load_test_tokenizer(),
        compute_metrics=compute_metric,
    )
    return test_trainer

test_dataset = preprocess_test_dataset("/content/drive/MyDrive/Colab Notebooks/Sarcasm/dataset/isarcasm/preprocessed/test.csv")
tokenized_test_data = tokenize_test_dataset(test_dataset)
# print(tokenized_data['train'][:2])
test_trainer = set_test_trainer(tokenized_test_data)
# trainer.train()
predictions = test_trainer.predict(tokenized_test_data)
f1_metric = evaluate.load("f1")
preds = np.argmax(predictions.predictions,axis=-1)
results = f1_metric.compute(predictions=preds, references=predictions.label_ids, average = "micro")
result_report = classification_report(preds,predictions.label_ids, target_names = class_names )
print(result_report)



from sklearn.metrics import confusion_matrix

# get predictions and true labels from trainer output
preds = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids

# create confusion matrix
cm = confusion_matrix(true_labels, preds)

# print confusion matrix
print("Confusion Matrix:")
print(cm)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# calculate the confusion matrix
cm = confusion_matrix(true_labels, preds)

# define class names
class_names = ['Non Sarcastic', 'Sarcastic']

# plot the confusion matrix
plt.matshow(cm)
plt.colorbar()
plt.xticks(range(len(class_names)), class_names)
plt.yticks(range(len(class_names)), class_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

