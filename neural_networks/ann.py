from sklearn.model_selection import train_test_split

class ANN():
    def __init__(self, data, config):
        self._prepare_data(data)

    def _prepare_data(self, data):
        X = data['Subject']
        y = data['Tray']

        print(f'y = {y}')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)