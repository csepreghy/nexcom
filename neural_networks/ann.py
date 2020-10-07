import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras.layers import Conv1D, Flatten, MaxPooling1D

class ANN():
    def __init__(self, config):
        self.config = config

    def _prepare_data(self, data):
        X = data['Subject']
        y = np.array(data['Tray']).reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.enc.fit(y_train)

        y_train = self.enc.transform(y_train)
        y_test = self.enc.transform(y_test)

        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        # print(self.enc.categories_)
        
        self.n_labels = y_train.shape[1]
        print(f'Number of labels: {self.n_labels}')

        return X_train, X_test, y_train, y_test

    def _build_model(self, config):
        model = Sequential()
        model.add(Embedding(config.vocab_size,
                            config.embedding_dims,
                            input_length=config.maxlen))
        model.add(Dropout(0.5))
        model.add(Conv1D(config.filters,
                        config.kernel_size,
                        padding='valid',
                        activation='relu'))
        model.add(MaxPooling1D())
        model.add(Conv1D(config.filters,
                        config.kernel_size,
                        padding='valid',
                        activation='relu'))

        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(config.hidden_dims, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_labels, activation='sigmoid'))

        return model
    
    def fit(self, data):
        X_train, X_test, y_train, y_test = self._prepare_data(data)
        model = self._build_model(self.config)
        # print(model.summary())

        # sparse_categorical_crossentropy
        model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

        model.fit(X_train, y_train,
                  batch_size=self.config.batch_size,
                  epochs=self.config.epochs,
                  validation_data=(X_test, y_test), callbacks=[])
