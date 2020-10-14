import numpy as np
import pandas as pd
import time
import datetime
import pickle
import openpyxl

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

        X_train, y_train = X, y

        tokenizer = text.Tokenizer(num_words=self.config.vocab_size)
        tokenizer.fit_on_texts(X_train)

        X_train = tokenizer.texts_to_matrix(X_train)
        X_train = sequence.pad_sequences(X_train, maxlen=self.config.maxlen)
        
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.enc.fit(y_train)

        y_train = self.enc.transform(y_train).toarray()

        print(self.enc.categories_)
        
        self.n_labels = y_train.shape[1]
        print(f'Number of labels: {self.n_labels}')

        X_train = np.expand_dims(X_train, axis=2)


        return X_train, y_train
    
    def predict(self, df_train, df_test, model):
        y_train = np.array(df_train['Tray']).reshape(-1, 1)
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(y_train)

        n_labels = y_train.shape[1]
        print('self.enc.categories_', enc.categories_)
        print(f'Number of labels: {n_labels}')

        X = df_test['Subject']

        tokenizer = text.Tokenizer(num_words=self.config.vocab_size)
        tokenizer.fit_on_texts(X)
    
        X = tokenizer.texts_to_matrix(X)
        X = sequence.pad_sequences(X, maxlen=self.config.maxlen)
        
        X_pred = np.expand_dims(X, axis=2)

        y_pred = model.predict(X_pred)
        print(f'y_pred = {y_pred}')

        predicted_labels = enc.inverse_transform(y_pred)
        df_test['Tray'] = predicted_labels

        print(f'{df_test}')

        df_test.to_excel('data/results.xlsx')



    def _build_model(self, config, n_labels=None):
        model = Sequential()

        model.add(Embedding(config.vocab_size, config.embedding_dims, input_length=config.maxlen))

        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))

        model.add(Dropout(0))
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.4))
        
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.4))

        if n_labels is not None:
            model.add(Dense(n_labels, activation='softmax'))

        else:
            model.add(Dense(self.n_labels, activation='softmax'))

        model.compile(loss=config.lossfunc, optimizer=Adam(0.001), metrics=['accuracy'])

        print(model.summary())

        return model
    
    def _get_callbacks(self, config):
        now = datetime.datetime.now()
        earlystopping = EarlyStopping(monitor='val_loss', patience=6)
        modelcheckpoint = ModelCheckpoint(filepath=f'{config.logpath}' + '/model/cnn.epoch{epoch:02d}-val_loss_{val_loss:.2f}' + f'-labels-{self.n_labels}' + '.h5',
                                          monitor='val_loss',
                                          save_best_only=True)
        
        return [earlystopping, modelcheckpoint]

    def fit(self, X_train, y_train):
        model = self._build_model(self.config)

        callbacks = self._get_callbacks(self.config)

        history = model.fit(X_train, y_train,
                            batch_size=self.config.batch_size,
                            epochs=self.config.epochs,
                            validation_split=0.1,
                            verbose=1,
                            callbacks=[callbacks])
        

        accuracies = history.history['accuracy']
        print(f'accuracies : {accuracies}')

        return model
    
    def load(self, path, n_labels):
        model = self._build_model(self.config, n_labels)
        model.load_weights(path)

        return model
        
