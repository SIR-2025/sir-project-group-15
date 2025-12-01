from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

QUESTION_LIMIT = 6

        
if __name__ == "__main__":
    app.run(port=8080)