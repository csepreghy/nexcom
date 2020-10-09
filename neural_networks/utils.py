import numpy as np
import tensorflow as tf
import pickle

from sklearn.metrics import f1_score, recall_score, precision_score, classification_report

def shuffle_in_unison(a, b):
    print(f'a.shape = {a.shape}')
    print(f'b.shape = {b.shape}')

    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return np.array(a), np.array(b)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    _, train_acc = model.evaluate(X_train, y_train)
    _, test_acc = model.evaluate(X_test, y_test)


    y_pred = model.predict(X_test)
    y_pred = tf.one_hot(tf.math.argmax(y_test, axis=1), depth=y_pred.shape[1])

    # precision = precision_score(y_test, y_pred), average='weighted')
    # recall = recall_score(y_test, y_pred), average='weighted'
    # f1 = f1_score(y_test, y_pred, average='micro')

    with open('logs/classification_report.txt', mode='w') as f:
        print(classification_report(y_test, y_pred, digits=3), file=f)
    
    # print(classification_report(y_test, y_pred, digits=3))
    print(f'train: {train_acc}')
    print(f'test: {test_acc}')

def save_accuracies(history):
    with open('cnn_accuracies.pkl', 'wb') as f:
        accuracies = history.history['accuracy']
        print(f'accuracies : {accuracies}')
        pickle.dump(history.history['accuracy'], f)
        
    with open ('cnn_accuracies.pkl', 'rb') as f:
        accuracies = pickle.load(f)
        print(f'accuracies : {accuracies}')
