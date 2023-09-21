"""#Import Libraries"""

!pip install transformers

from sklearn.svm import SVC
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd

"""#Import Dataset"""

from google.colab import drive
drive.mount('/content/drive')

train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Sarcasm/dataset/isarcasm/clean/train.csv')
test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Sarcasm/dataset/isarcasm/clean/test.csv')

"""#BERT Embeddings"""

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings for text
def get_bert_embeddings(text):
    # Tokenize text
    input_ids = tokenizer.encode(text, return_tensors='pt')
    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(input_ids)
    # Return the last hidden state of the first token (CLS token)
    return outputs[0][:,0,:].numpy()

# Get BERT embeddings for train and test data
X_train = [get_bert_embeddings(text).reshape(-1) for text in train['tweet']]
X_test = [get_bert_embeddings(text).reshape(-1) for text in test['tweet']]

# Get labels for train data
y_train = train['sarcastic']

"""#SVM Classifier"""

# Create SVM classifier with rbf kernel and regularization value 10
clf = SVC(kernel='rbf', C=10)

# Fit classifier on train data
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

"""#Report"""

from sklearn.metrics import classification_report

# Get true labels for test data
y_test = test['sarcastic']

# Print classification report
print(classification_report(y_test, y_pred))