import config
from neural_networks.ann import ANN
from preprocessing import load_text, run_preprocessing
from utils import evaluate_model

if __name__ == '__main__':
    df = load_text(max_len=-1)
    ann = ANN(config)

    X_train, X_test, X_val, y_train, y_test, y_val = ann.prepare_data(df)
    model = ann.fit(X_train, X_test, X_val, y_train, y_test, y_val)
    train_acc, test_acc, f1 = evaluate_model(model, X_train, y_train, X_test, y_test)