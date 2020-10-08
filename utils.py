import numpy as np
import tensorflow as tf

from sklearn.metrics import f1_score

def evaluate_model(model, X_train, y_train, X_test, y_test):
    _, train_acc = model.evaluate(X_train, y_train)
    _, test_acc = model.evaluate(X_test, y_test)

    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    y_pred = model.predict(X_test)
    y_pred = tf.one_hot(tf.math.argmax(y_test, axis=1), depth=y_pred.shape[1])

    f1 = f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))

    print(f'Weighted f1 score: {f1}')

    return train_acc, test_acc, f1