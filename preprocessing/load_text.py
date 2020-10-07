from os import listdir
from os.path import isfile, join
import pandas as pd
import string
import re

def load_text(max_len):
    df_raw_data = pd.read_csv('data/Exercise STUDENT Training dataset - 1X IRI 7 Oct 20.csv').head(max_len)
    print(f'{df_raw_data}')

    return df_raw_data

def count_empty_subjects(df):
    total = 0
    for subject in df['Subject']:
        nospace = re.sub(r'(\s|\u180B|\u200B|\u200C|\u200D|\u2060|\uFEFF)+', '', str(subject))

        if len(nospace) == 0 or nospace == 'nan':
            total+=1

    return total

def run_preprocessing(df):
    subjects = df['Subject']
    n_empty_subjects = count_empty_subjects(df)
    print(n_empty_subjects)
    