import pandas as pd

from neural_networks.cnn import CNN
import config
from neural_networks.utils import evaluate_model
from preprocessing import load_text, run_preprocessing

if __name__ == '__main__':
    df = load_text(max_len=-1)
    cnn = CNN(config)
    X_train, X_test, y_train, y_test = cnn.prepare_data(df)

    model = cnn.load(path='logs/model/cnn.epoch10-val_loss_0.53-labels-95.h5',
                     n_labels=90)
    evaluate_model(model, X_train, y_train, X_test, y_test)
