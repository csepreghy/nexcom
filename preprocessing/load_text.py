from os import listdir
from os.path import isfile, join
import pandas as pd
import string
import re
import xlrd

def load_text(max_len):
    df = pd.read_excel('data/Exercise STUDENT Results dataset - 1X IRI 7 Oct 20.xlsx').astype(str)
    # df = pd.read_csv('data/Exercise STUDENT Training dataset - 1X IRI 7 Oct 20.csv').head(max_len).astype(str)
    print(f'{df}')
    print(df.Tray.value_counts().tolist())

    return df

def count_empty_subjects(df):
    total = 0
    for subject in df['Subject']:
        nospace = re.sub(r'(\s|\u180B|\u200B|\u200C|\u200D|\u2060|\uFEFF)+', '', str(subject))

        if len(nospace) == 0 or nospace == 'nan':
            total+=1

    return total


def print_plot(df, index):
    example = df[df.index == index][['Subject', 'Tray']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Tray:', example[1])

def run_preprocessing(df):
    subjects = df['Subject']
    n_empty_subjects = count_empty_subjects(df)
    print(n_empty_subjects)
    