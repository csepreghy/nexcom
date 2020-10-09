import numpy as np
import pandas as pd
import time
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (Embedding,
                                     Conv1D,
                                     Flatten,
                                     MaxPooling1D,
                                     Dense,
                                     Dropout,
                                     Activation,
                                     SpatialDropout1D,
                                     LSTM,
                                     CuDNNLSTM)
from tensorflow.keras import layers
from tensorflow.keras.callbacks import History, TensorBoard, EarlyStopping, ModelCheckpoint

from .utils import shuffle_in_unison

class LongShortTermMemory():
    def __init__(self, config):
        self.config = config

    def prepare_data(self, data):
        X = data['Subject']
        y = np.array(data['Tray']).reshape(-1, 1)

        X, y = shuffle_in_unison(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        tokenizer = text.Tokenizer(num_words=self.config.vocab_size)
        tokenizer.fit_on_texts(X_train)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)
        X_val = tokenizer.texts_to_sequences(X_val)

        X_train = sequence.pad_sequences(X_train, maxlen=self.config.maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=self.config.maxlen)
        X_val = sequence.pad_sequences(X_val, maxlen=self.config.maxlen)
        
        print('Shape of data tensor:', X_train.shape)

        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.enc.fit(y_train)

        y_train = self.enc.transform(y_train).toarray()
        y_test = self.enc.transform(y_test).toarray()
        y_val = self.enc.transform(y_val).toarray()

        # print(self.enc.categories_)
        
        self.n_labels = y_train.shape[1]
        print(f'Number of labels: {self.n_labels}')

        print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
        print(f'X_test: {X_test.shape}, y_test: {y_test.shape}')

        return X_train, X_test, X_val, y_train, y_test, y_val

    def _build_cpu_model(self, config):
        model = Sequential()
        model.add(Embedding(config.maxlen, config.embedding_dims, input_length=config.maxlen))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(self.n_labels, activation='softmax'))
        
        model.compile(loss=config.lossfunc, optimizer=Adam(0.0001), metrics=['accuracy'])
        print(model.summary())

        return model

    def _build_gpu_model(self, config):
        model = Sequential()
        model.add(Embedding(config.maxlen, config.embedding_dims, input_length=config.maxlen))
        model.add(SpatialDropout1D(0.2))
        model.add(CuDNNLSTM(128)) # recurrent_dropout=0.2
        model.add(Dropout(0.2))
        model.add(CuDNNLSTM(128))
        model.add(Dropout(0.2))

        model.add(Dense(self.n_labels, activation='softmax'))
        
        model.compile(loss=config.lossfunc, optimizer=Adam(0.0001), metrics=['accuracy'])
        print(model.summary())

        return model
    
    def _get_callbacks(self, config):
        now = datetime.datetime.now()
        tensorboard = TensorBoard(log_dir=f'{config.logpath}/lstm-{now}')
        earlystopping = EarlyStopping(monitor='accuracy', patience=10)
        modelcheckpoint = ModelCheckpoint(filepath=f'{config.logpath}' + '/model/lstm.epoch{epoch:02d}-val_loss_{val_loss:.2f}.h5',
                                          monitor='val_loss',
                                          save_best_only=True)
        
        return [tensorboard, earlystopping, modelcheckpoint]

    def fit(self, X_train, X_test, X_val, y_train, y_test, y_val):
        model = self._build_model(self.config)

        callbacks = self._get_callbacks(self.config)
        print(X_train.shape)
        print(X_train[0].shape)

        history = model.fit(X_train, y_train,
                            batch_size=self.config.batch_size,
                            epochs=self.config.epochs,
                            validation_data=(X_val, y_val),
                            callbacks=[callbacks])

        return model
        
