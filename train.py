import config
from neural_networks.cnn import CNN
from neural_networks.lstm import LSTM
from preprocessing import load_text, run_preprocessing
from neural_networks.utils import evaluate_model

def run_cnn(df):
    cnn = CNN(config)
    X_train, X_test, X_val, y_train, y_test, y_val = cnn.prepare_data(df)
    model = cnn.fit(X_train, X_test, X_val, y_train, y_test, y_val)
    evaluate_model(model, X_train, y_train, X_test, y_test)

def run_lstm(df):
    lstm = LSTM(config)
    X_train, X_test, X_val, y_train, y_test, y_val = lstm.prepare_data(df)
    model = lstm.fit(X_train, X_test, X_val, y_train, y_test, y_val)
    evaluate_model(model, X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    df = load_text(max_len=-1)
    # run_cnn(df)
    run_lstm(df)
    