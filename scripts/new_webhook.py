from flask import Flask, request, jsonify
import pandas as pd
import re

app = Flask(__name__)

# DATA LOADING 
def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    y = df.iloc[:, 0] 
    X = df.iloc[:, 1:]
    return X, y

dataset_path = "animal_dataset.csv"
X, y = load_dataset(dataset_path)

# SESSION STATE STORAGE
session_states = {}

def get_session_id(session_str):
    match = re.search(r'sessions/([^/]+)$', session_str)
    if match:
        return match.group(1)
    return session_str

def init_state(session_id):
    session_states[session_id] = {
        "likelihoods": pd.Series(0.0, index=X.index).to_dict(),
        "asked_features": [],
        "turn_count": 0,
        "last_feature_asked": None,
        "pending_guess_animal": None
    }

# LOGIC

def best_question(new_series, X, asked_features): 
    best_likelihoods = new_series[new_series == new_series.max()].index 
    
    X_sub = X.loc[best_likelihoods].drop(columns=asked_features, errors='ignore') 
    
    best_feature = None 
    best_split = 1.0 
    
    if X_sub.empty:
        # we ran out of features for the top candidates, pick any remaining feature from the whole dataset
        remaining = [c for c in X.columns if c not in asked_features]
        return remaining[0] if remaining else None

    for feature in X_sub.columns: 
        yes_ratio = (X_sub[feature] == 1).mean() 
        split_quality = abs(0.5 - yes_ratio) 
        if split_quality < best_split: 
            best_split = split_quality 
            best_feature = feature 
    return best_feature

def update_likelihood(animal_value, current_val, answer):
    if answer in ('yes', 'y', 'true'):
        return current_val + (1 if animal_value == 1 else 0)
    elif answer in ('probably', 'probably yes'):
        return current_val + (0.75 if animal_value == 1 else 0.25)
    if answer in ('i dont know', 'idk', 'maybe', 'i don\'t know'):
        return current_val + 0.5
    if answer in ('probably not', 'probably no'):
        return current_val + (0.25 if animal_value == 1 else 0.75)
    if answer in ('no', 'n', 'false'):
        return current_val + (0 if animal_value == 1 else 1)
    return current_val

# WEBHOOK ENDPOINT

@app.route("/", methods=["POST"])
def webhook():
    req = request.get_json()
    session_path = req.get("sessionInfo", {}).get("session", "default")
    session_id = get_session_id(session_path)
    question_limit = 6
    
    # Get User Answer
    params = req.get("sessionInfo", {}).get("parameters", {})
    user_answer = params.get("last_answer", "start").lower().strip()

    if session_id not in session_states or user_answer == "reset":
        init_state(session_id)
        user_answer = None

    state = session_states[session_id]
    
    new_series = pd.Series(state["likelihoods"])
    response_text = ""
    
    # UPDATE STATE BASED ON PREVIOUS ANSWER
    
    if state["pending_guess_animal"]:
        if user_answer in ('yes', 'y', 'correct'):
            response_text = "yipie, I got it correct! Say 'reset' to play again."

            del session_states[session_id]
            return jsonify({"fulfillment_response": {"messages": [{"text": {"text": [response_text]}}]}})
        else:
            # Find the index of the animal we just guessed and remove it.
            guessed_idx = y[y == state["pending_guess_animal"]].index[0]
            new_series = new_series.drop(guessed_idx)
            
            response_text = "Okay, not that. Let me think... "
            state["pending_guess_animal"] = None 

    elif state["last_feature_asked"] and user_answer:
        feature = state["last_feature_asked"]
        # Update scores for every animal
        for i in new_series.index:
            val = X.loc[i, feature]
            # Calculate new score
            new_series.loc[i] = update_likelihood(val, new_series.loc[i], user_answer)
        
        state["asked_features"].append(feature)
        state["turn_count"] += 1
        state["last_feature_asked"] = None

    # DECIDE NEXT MOVE (ask question vs guess animal logic)
    
    state["likelihoods"] = new_series.to_dict()
    
    
    # 1: THE FIRST 6 QUESTIONS
    if state["turn_count"] < question_limit:
        feature = best_question(new_series, X, state["asked_features"])
        if feature:
            state["last_feature_asked"] = feature
            response_text += f"{feature}?"
        else:
            # No features? Skip to guessing
            state["turn_count"] = question_limit
            
    # 2: THE GUESSING LOGIC
    if state["turn_count"] >= question_limit:
        
        if new_series.empty:
             response_text = "I have run out of animals! Say 'reset' to try again."
             return jsonify({"fulfillment_response": {"messages": [{"text": {"text": [response_text]}}]}})

        best_likelihoods = new_series[new_series == new_series.max()].index
        
        # TIE (more than 1 top candidate)
        if len(best_likelihoods) > 1:
            feature = best_question(new_series, X, state["asked_features"])
            
            if feature:
                state["last_feature_asked"] = feature
                response_text += f"{feature}?"
            else:
                # If no questions left to split the tie, just guess the first one
                animal = y[best_likelihoods[0]]
                state["pending_guess_animal"] = animal
                response_text += f"I think your animal is a {animal}?"

        # SINGLE WINNER
        else:
            animal = y[best_likelihoods[0]]
            state["pending_guess_animal"] = animal
            response_text += f"I think your animal is a {animal}?"

    return jsonify({
        "fulfillment_response": {
            "messages": [{"text": {"text": [response_text]}}]
        }
    })

if __name__ == "__main__":
    app.run(port=8080)