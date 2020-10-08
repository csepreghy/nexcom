import config
from neural_networks.cnn import CNN
from preprocessing import load_text, run_preprocessing
from utils import evaluate_model

if __name__ == '__main__':
    df = load_text(max_len=-1)
    cnn = CNN(config)

    X_train, X_test, X_val, y_train, y_test, y_val = cnn.prepare_data(df)
    model = cnn.fit(X_train, X_test, X_val, y_train, y_test, y_val)
    evaluate_model(model, X_train, y_train, X_test, y_test)