from os import listdir
from os.path import isfile, join

def load_text():
    mypath = 'data/bbc'
    filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # for f in files:
    #     with open(f, 'r') as reader:
    #         print(reader.read())

    return filenames