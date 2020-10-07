from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras.layers import Conv1D, Flatten, MaxPooling1D

class ANN():
    def __init__(self, data, config):
        self._prepare_data(data)

    def _prepare_data(self, data):
        X = data['Subject']
        y = data['Tray']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
        model.add(Dense(1, activation='sigmoid'))

        return model
    
    def fit(self, config):
        model = self._build_model(config)

