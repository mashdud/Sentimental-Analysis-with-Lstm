from read_data import *
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, SpatialDropoout1D, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split



def clean_text(text,REPLACE_BY_SPACE_RE, BAD_SYMBOLS_RE, STOPWORDS):
    """
    text: a string
    return: modified initial string
    """
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

if __name__ == '__main__':
  # read and preprocess train dataset
  train = read_data("../data/drugsComTrain_raw.tsv")
  train['review'] = train['review'].apply(clean_text)
  # The maximum number of words to be used. (most frequent)
  MAX_NB_WORDS = 10000
  # max number of words in each review
  MAX_SEQUENCE_LENGTH = 200
  # this is fixed
  EMBEDDING_DIM = 100
  tokenizer = Tokenizer(num_words = MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
  tokenizer.fit_on_texts(train['review'].values)
  word_index = tokenizer.word_index
  X = tokenizer.texts_to_sequence(train['review'].values)
  X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
  y = pd.get_dummies(train['rating']).values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  # model
  model = Sequential()
  model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
  model.add(SpatialDropoout1D(0.2))
  model.add(LSTM(100, dropout=0.2, recurrent_dropout=0,return_sequences=True))
  model.add(LSTM(100, dropout=0.2, recurrent_dropout=0))
  model.add(Dense(10, activation='softmax'))
  optimizer = keras.optimizers.Adam(lr=0.0001)
  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  epochs = 10
  batch_size = 64
  history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
  print(accr = model.evaluate(X_test, y_test))









