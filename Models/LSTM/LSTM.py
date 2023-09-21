"""## Pre-Processed Dataset"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv('/content/drive/MyDrive/dataset/Sarcasm Detection/isarcasm/clean/train.csv')
test_data = pd.read_csv('/content/drive/MyDrive/dataset/Sarcasm Detection/isarcasm/clean/test.csv')

# rename the columns to 'text' and 'label'
train_data = train_data.rename(columns={'tweet': 'text', 'sarcastic': 'label'})
test_data = test_data.rename(columns={'tweet': 'text', 'sarcastic': 'label'})

"""## Long Short-Term Memory (LSTM) Implementation
LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) architecture that is particularly effective for sequence data like text. The basic idea behind LSTMs is to have a memory cell that can store information over time and selectively forget or remember that information as needed. This allows LSTMs to model long-term dependencies in the input sequence, which is difficult for other types of neural networks to do.

The memory cell in an LSTM has three gates: the input gate, the forget gate, and the output gate. The input gate controls whether new input should be added to the memory cell, the forget gate controls whether old information should be forgotten from the memory cell, and the output gate controls how much of the current memory cell state should be output. Each gate is implemented using a sigmoid activation function that outputs a value between 0 and 1, which controls how much information should be passed through.
"""

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, TimeDistributed
from sklearn.metrics import classification_report

"""Tokenization is the process of converting raw text data into numerical sequences that can be fed into a neural network. The Keras Tokenizer class is used to perform this task. First, the tokenizer is instantiated with the maximum number of words to keep in the vocabulary, which is set to 10,000 in this case. Then, the fit_on_texts method is called on the training data to fit the tokenizer on the text data and generate the vocabulary. Finally, the texts_to_sequences method is used to convert the text data into sequences of integers.

By setting a limit on the number of words in the vocabulary, we can ensure that the model focuses on the most important words in the dataset and ignores less frequent words that may not be useful for classification. In addition, limiting the size of the vocabulary also helps to prevent overfitting, as it reduces the number of parameters that need to be learned by the model.
"""

# Tokenize the data
max_features = 10000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(train_data['text'])
X = tokenizer.texts_to_sequences(train_data['text'])
X = pad_sequences(X)

"""The LSTM model is defined using the Keras Sequential class, which allows us to stack layers on top of each other. The first layer is an Embedding layer that learns a dense representation of the input text. The Embedding layer takes the vocabulary size (10,000) and the embedding dimension (128) as input. The second layer is an LSTM layer with 128 units and a dropout rate of 0.2. The dropout rate is used to prevent overfitting by randomly dropping out units during training. The final layer is a Dense layer with a single unit and a sigmoid activation function, which outputs a binary classification result."""

# Model Architecture
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(TimeDistributed(Dense(128, activation='relu')))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(TimeDistributed(Dense(128, activation='relu')))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

"""The model is trained on the training data using the binary cross-entropy loss function, the Adam optimizer, and the accuracy metric. Binary cross-entropy loss is used because we are performing binary classification (sarcasm vs non-sarcasm). The Adam optimizer is a popular optimization algorithm that uses adaptive learning rates to converge faster. During training, the data is split into a training set and a validation set, with 20% of the data used for validation. The model is trained for 5 epochs with a batch size of 32."""

# Train the model
model.fit(X, train_data['label'], batch_size=32, epochs=10, validation_split=0.2)

"""Once the model is trained, it is used to make predictions on the test data. The test data is first tokenized using the same tokenizer that was fit on the training data. Then, the pad_sequences method is used to ensure that all sequences have the same length. Finally, the predict method is called on the model to generate predictions for the test set. The predictions are continuous values between 0 and 1, so we round them to the nearest integer to obtain binary labels."""

# Predict on test data
test_sequences = tokenizer.texts_to_sequences(test_data['text'])
test_sequences = pad_sequences(test_sequences, maxlen=X.shape[1])
y_pred = model.predict(test_sequences)

# Convert predictions to labels
y_pred = np.round(y_pred).astype(int)

"""The classification_report function from the scikit-learn library is used to compute the F1 score and other classification metrics for the test set. The classification report shows the precision, recall, F1 score, and support for each class (sarcasm and non-sarcasm), as well as the weighted averages for each metric. The F1 score is a measure of the model's accuracy that takes into account both precision and recall. A higher F1 score indicates better overall performance."""

# Evaluate the model
print(classification_report(test_data['label'], y_pred))