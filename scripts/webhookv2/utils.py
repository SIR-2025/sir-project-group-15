import pandas as pd
import re
from flask import jsonify
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ANIMAL_DATASET_PATH = os.path.join(BASE_DIR, "animal_dataset.csv")

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

def jsonify_response(response_text):
    return jsonify({
        "fulfillment_response": {
            "messages": [{"text": {"text": [response_text]}}]
        }
    })
    
    