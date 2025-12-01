import pandas as pd

def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    y = df.iloc[:, 0]        
    X = df.iloc[:, 1:]     
    return X, y