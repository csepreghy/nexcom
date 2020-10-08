import config
from neural_networks.ann import ANN
from preprocessing import load_text, run_preprocessing

if __name__ == '__main__':
    df = load_text(max_len=-1)
    ann = ANN(config)
    ann.fit(df)