from flask import Flask

app = Flask(__name__)

from utils import get_session_id, jsonify_response, ANIMAL_DATASET_PATH
from models import GuessingGameModel

from flask import request

models = {}

@app.route("/", methods=["POST"])
def webhook():
    req = request.get_json()
    session_path = req.get("sessionInfo", {}).get("session", "default")
    session_id = get_session_id(session_path)
    
    # Get User Answer
    params = req.get("sessionInfo", {}).get("parameters", {})
    user_answer = params.get("last_answer", "start").lower().strip()

    if session_id not in models or user_answer == "reset":
        models[session_id] = GuessingGameModel(session_id, ANIMAL_DATASET_PATH)
        user_answer = None

    model = models[session_id]
    response_text = model.next_response(user_answer)

    return jsonify_response(response_text)

        
if __name__ == "__main__":
    app.run(port=8080)