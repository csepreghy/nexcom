import pandas as pd

from neural_networks.cnn import CNN
import config
from neural_networks.utils import evaluate_model
from preprocessing import load_text, run_preprocessing

if __name__ == '__main__':
    df_train = pd.read_csv('data/Exercise STUDENT Training dataset - 1X IRI 7 Oct 20.csv').astype(str)
    df_test = pd.read_excel('data/Exercise STUDENT Results dataset - 1X IRI 7 Oct 20.xlsx').astype(str)
    cnn = CNN(config)
    # X_train, X_test, y_train, y_test = cnn.prepare_data(df)

    model = cnn.load(path='logs/model/cnn.epoch08-val_loss_0.28-labels-96.h5',
                     n_labels=96)
    
    cnn.predict(df_train, df_test, model)

    # evaluate_model(model, X_train, y_train, X_test, y_test)
