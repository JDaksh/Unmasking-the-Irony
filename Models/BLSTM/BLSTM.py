"""## Pre-Processed Dataset"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('/content/drive/MyDrive/dataset/Sarcasm Detection/isarcasm/clean/train.csv')
test_data = pd.read_csv('/content/drive/MyDrive/dataset/Sarcasm Detection/isarcasm/clean/test.csv')

print('Number of instances = %d' % (train_data.shape[0]))
print('Number of attributes = %d' % (train_data.shape[1]))

# rename the columns to 'text' and 'label'
train_data = train_data.rename(columns={'tweet': 'text', 'sarcastic': 'label'})
test_data = test_data.rename(columns={'tweet': 'text', 'sarcastic': 'label'})
train_data.head()

train_data['label'].value_counts()

sns.countplot(x='label', data=train_data)
plt.show()

"""## Bidirectional Long Short-Term Memory (BLSTM) Implementation

"""

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, TimeDistributed
from sklearn.metrics import classification_report

def train_model(X_train, y_train, X_test, y_test, max_tweet_len):
    # Tokenize the training set
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Pad the sequences
    X_train = pad_sequences(X_train, maxlen=max_tweet_len, padding='post')
    X_test = pad_sequences(X_test, maxlen=max_tweet_len, padding='post')

    # Define the vocabulary size and embedding dimension
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100

    # Define model architecture
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_tweet_len))

    # First BLSTM layer followed by a time-distributed dense layer
    model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(TimeDistributed(Dense(units=32, activation='relu')))

    # Second BLSTM layer followed by a time-distributed dense layer
    model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(TimeDistributed(Dense(units=32, activation='relu')))

    # Third BLSTM layer followed by two dense layers
    model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred = [1 if y >= 0.5 else 0 for y in y_pred]

    # Print classification report
    print(classification_report(y_test, y_pred))

# Train and evaluate model
train_model(train_data['text'], train_data['label'], test_data['text'], test_data['label'], max_tweet_len=100)