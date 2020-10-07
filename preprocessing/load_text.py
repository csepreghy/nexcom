from os import listdir
from os.path import isfile, join
import pandas as pd

import preprocessing.config as config

def load_text():
    df_raw_data = pd.read_csv('data/Exercise STUDENT Training dataset - 1X IRI 7 Oct 20.csv')
    print(f'{df_raw_data}')

    return 0