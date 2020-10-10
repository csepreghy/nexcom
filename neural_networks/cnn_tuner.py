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
from kerastuner.tuners import RandomSearch
import tensorflow as tf

from .utils import shuffle_in_unison, save_accuracies

class CNNTuner():
    def __init__(self, config):
        self.config = config

    def prepare_data(self, data):
        X = data['Subject']
        y = np.array(data['Tray']).reshape(-1, 1)

        print(f'X[0:5] = {X[0:5]}')
        X, y = shuffle_in_unison(X, y)
        print(f'X[0:5] = {X[0:5]}')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        tokenizer = text.Tokenizer(num_words=self.config.vocab_size)
        tokenizer.fit_on_texts(X_train)

        X_train = tokenizer.texts_to_matrix(X_train)
        X_test = tokenizer.texts_to_matrix(X_test)
        X_val = tokenizer.texts_to_matrix(X_val)

        X_train = sequence.pad_sequences(X_train, maxlen=self.config.maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=self.config.maxlen)
        X_val = sequence.pad_sequences(X_val, maxlen=self.config.maxlen)
        
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.enc.fit(y_train)

        y_train = self.enc.transform(y_train).toarray()
        y_test = self.enc.transform(y_test).toarray()
        y_val = self.enc.transform(y_val).toarray()

        # print(self.enc.categories_)
        
        self.n_labels = y_train.shape[1]
        print(f'Number of labels: {self.n_labels}')

        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)
        X_val = np.expand_dims(X_val, axis=2)

        return X_train, X_test, X_val, y_train, y_test, y_val

    def _build_model(self, hp):
        config = self.config
        model = Sequential()

        model.add(Embedding(config.vocab_size, config.embedding_dims, input_length=config.maxlen))

        for i in range(hp.Int('n_conv_layers', 1, 5)):
            model.add(Conv1D(filters=hp.Choice('n_filters', values=[32, 64, 128, 256, 512]), kernel_size=3, activation='relu'))
            model.add(Dropout(0.1))
            if i < 3: model.add(MaxPooling1D(pool_size=2))
        
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(Dropout(0.1))

        model.add(Flatten())

        for i in range(hp.Int('n_dense_layers', 1, 3)):
            model.add(Dense(hp.Choice('n_dense_neurones', values=[32, 64, 128, 256, 512]), activation='relu'))
            model.add(Dropout(0.5))

        model.add(Dense(self.n_labels, activation='softmax'))
        model.compile(loss=config.lossfunc, optimizer=Adam(hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001, 0.00001])), metrics=['accuracy'])

        # print(model.summary())

        return model
    
    def fit(self, X_train, X_test, X_val, y_train, y_test, y_val):
        tuner = RandomSearch(self._build_model,
                             objective='val_accuracy',
                             max_trials=self.config.max_tuner_trials,
                             executions_per_trial=1,
                             directory='logs/keras-tuner/',
                             project_name='cnn-tuner')

        tuner.search_space_summary()
    
        tuner.search(x=X_train,
                     y=y_train,
                     epochs=self.config.epochs,
                     batch_size=self.config.batch_size,
                     verbose=2,
                     validation_data=(X_val, y_val),
                     callbacks=[EarlyStopping('val_accuracy', patience=6)])
        
        print(tuner.results_summary())
        model = tuner.get_best_models(num_models=1)[0]
        print(model.summary())


        return model
        
