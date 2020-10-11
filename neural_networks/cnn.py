import numpy as np
import pandas as pd
import time
import datetime
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, Flatten, MaxPooling1D, Dense, Dropout, Activation
from tensorflow.keras.callbacks import History, TensorBoard, EarlyStopping, ModelCheckpoint

from .utils import shuffle_in_unison

class CNN():
    def __init__(self, config):
        self.config = config

    def prepare_data(self, data):
        X = data['Subject']
        y = np.array(data['Tray']).reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        tokenizer = text.Tokenizer(num_words=self.config.vocab_size)
        tokenizer.fit_on_texts(X_train)

        X_train = tokenizer.texts_to_matrix(X_train)
        X_test = tokenizer.texts_to_matrix(X_test)
        X_val = tokenizer.texts_to_matrix(X_val)

        X_train = sequence.pad_sequences(X_train, maxlen=self.config.maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=self.config.maxlen)
        
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.enc.fit(y_train)

        y_train = self.enc.transform(y_train).toarray()
        y_test = self.enc.transform(y_test).toarray()

        # print(self.enc.categories_)
        
        self.n_labels = y_train.shape[1]
        print(f'Number of labels: {self.n_labels}')

        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)

        return X_train, X_test, y_train, y_test

    def _build_model(self, config):
        model = Sequential()

        model.add(Embedding(config.vocab_size, config.embedding_dims, input_length=config.maxlen))

        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(Dropout(0.2))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.6))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.6))

        model.add(Dense(self.n_labels, activation='softmax'))
        model.compile(loss=config.lossfunc, optimizer=Adam(0.0001), metrics=['accuracy'])

        # print(model.summary())

        return model
    
    def _get_callbacks(self, config):
        now = datetime.datetime.now()
        earlystopping = EarlyStopping(monitor='accuracy', patience=15)
        modelcheckpoint = ModelCheckpoint(filepath=f'{config.logpath}' + '/model/cnn.epoch{epoch:02d}-val_loss_{val_loss:.2f}.h5',
                                          monitor='val_loss',
                                          save_best_only=True)
        
        return [earlystopping, modelcheckpoint]

    def fit(self, X_train, X_test, X_val, y_train, y_test):
        model = self._build_model(self.config)

        callbacks = self._get_callbacks(self.config)

        history = model.fit(X_train, y_train,
                            batch_size=self.config.batch_size,
                            epochs=self.config.epochs,
                            validation_split=0.2,
                            verbose=2,
                            callbacks=[callbacks])
        

        accuracies = history.history['accuracy']
        print(f'accuracies : {accuracies}')

        return model
        
