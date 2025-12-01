from utils import get_session_id

from flask import request, jsonify
        

@app.route("/", methods=["POST"])
def webhook():
    req = request.get_json()
    session_path = req.get("sessionInfo", {}).get("session", "default")
    session_id = get_session_id(session_path)
    
    # Get User Answer
    params = req.get("sessionInfo", {}).get("parameters", {})
    user_answer = params.get("last_answer", "start").lower().strip()

    if session_id not in session_states or user_answer == "reset":
        init_state(session_id)
        user_answer = None

    state = session_states[session_id]
    
    user_answer = parse_user_answer(state, user_answer)
    response_text = get_next_question(state, user_answer)

    return jsonify({
        "fulfillment_response": {
            "messages": [{"text": {"text": [response_text]}}]
        }
    })
