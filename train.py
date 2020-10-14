import config
from neural_networks.cnn import CNN
from neural_networks.cnn_tuner import CNNTuner
from neural_networks.lstm import LongShortTermMemory
from preprocessing import load_text, run_preprocessing
from neural_networks.utils import evaluate_model

def run_cnn(df):
    cnn = CNN(config)
    X_train, y_train = cnn.prepare_data(df)
    model = cnn.fit(X_train, y_train)
    # evaluate_model(model, X_train, y_train, X_test, y_test)

def run_lstm(df):
    lstm = LongShortTermMemory(config)
    X_train, X_test, X_val, y_train, y_test, y_val = lstm.prepare_data(df)
    model = lstm.fit(X_train, X_test, X_val, y_train, y_test, y_val)
    evaluate_model(model, X_train, y_train, X_test, y_test)

def run_cnn_tuner(df):
    cnn_tuner = CNNTuner(config)
    X_train, X_test, y_train, y_test = cnn_tuner.prepare_data(df)
    model = cnn_tuner.fit(X_train, X_test, y_train, y_test)
    evaluate_model(model, X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    df = load_text(max_len=-1)
    run_cnn(df)
    # run_cnn_tuner(df)
    # run_lstm(df)
    