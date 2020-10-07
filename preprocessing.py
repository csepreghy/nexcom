from preprocessing import load_text, run_preprocessing


if __name__ == '__main__':
    df = load_text()
    processed = run_preprocessing(df)