from utils import get_session_id, jsonify_response
from models import GuessingGameModel

from flask import request

models = {}
ANIMAL_DATASET_PATH = "animal_dataset.csv"

@app.route("/", methods=["POST"])
def webhook(models, animal_dataset_path="animal_dataset.csv"):
    req = request.get_json()
    session_path = req.get("sessionInfo", {}).get("session", "default")
    session_id = get_session_id(session_path)
    
    # Get User Answer
    params = req.get("sessionInfo", {}).get("parameters", {})
    user_answer = params.get("last_answer", "start").lower().strip()

    if session_id not in models or user_answer == "reset":
        models[session_id] = GuessingGameModel(session_id, animal_dataset_path)
        user_answer = None

    model = models[session_id]
    response_text = model.next_response(user_answer)

    return jsonify_response(response_text)
