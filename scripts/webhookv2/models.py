from flask import jsonify

def parse_user_response(state, user_answer, new_series):
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