from neural_networks.ann import ANN
from preprocessing import load_text, run_preprocessing
import config

if __name__ == '__main__':
    df = load_text(max_len=500)
    ann = ANN(df, config)