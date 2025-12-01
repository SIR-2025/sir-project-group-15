import pandas as pd
import re

def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    y = df.iloc[:, 0]        
    X = df.iloc[:, 1:]     
    return X, y

def get_session_id(session_str):
    match = re.search(r'sessions/([^/]+)$', session_str)
    if match:
        return match.group(1)
    return session_str
    