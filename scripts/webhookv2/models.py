from guessing_algorithm import *
from utils import load_dataset
from responses import *

import pandas as pd
from threading import Thread
import random

class GuessingGameModel:
    first_question_counter = 0
    global_asked_features = set()

    def __init__(self, session_id, animal_dataset_path):
        self.session_id = session_id

        options = [0, 8, 20]
        self.first_feature_index = options[GuessingGameModel.first_question_counter % 3]
        GuessingGameModel.first_question_counter += 1

        self.X, self.y = load_dataset(animal_dataset_path)
        self.likelihood = pd.Series(0.0, index=self.X.index)
        self.asked_features = set()
        self.useless_features = set()
        self.turn_count = 0
        self.question_number = 0
        self.last_feature_asked = None
        self.pending_guess_animal = None
        self.top_history = []
        self.awaiting_user_animal = False
        self.answer_history = {}
        self.MIN_ASKED_FEATURES_BEFORE_STUCK_CHECK = 4

        with open('extra_text.txt', 'r', encoding='utf-8') as f:
            self.extra_text = f.read().splitlines()

    def next_response(self, user_answer):
        response_text = ""

        # 1. If waiting for confirmation of animal
        if self.awaiting_user_animal:
            animal_name = user_answer.strip()
            self.awaiting_user_animal = False
            self.reset_game()
            return f"Thanks! I am sorry I could not guess. Do you want to play again?."
        
        # 2. If last turn was a guess...
        if self.pending_guess_animal:
            guessed_right, response_text = self.validate_guess(user_answer)
            if guessed_right:
                self.reset_game()
                return response_text

        # 3. Update likelihood
        if self.last_feature_asked is not None:
            feature = self.last_feature_asked
            for i in self.likelihood.index:
                self.likelihood = update_likelihood(
                    self.X.loc[i, feature],
                    self.likelihood,
                    i,
                    user_answer
                )
            self.last_feature_asked = None
            self.track_top_animals()

        # 4. Check if confident enough
        if is_confident_enough(self.likelihood):
            best_idx = self.likelihood.idxmax()
            animal = self.y[best_idx]
            self.pending_guess_animal = animal
            return response_text + GUESS_RESPONSES.format(animal=animal)


        # 5. Otherwise pick next feature
        if len(self.asked_features) == 0:
            feature = self.X.columns[self.first_feature_index]
            response_text = ""
            got_stuck = False
            text_idx = self.first_feature_index
        else:
            feature, response_text, got_stuck, text_idx = self.get_next_feature()

        if got_stuck:
            final_text = response_text + START_OVER_RESPONSE
            self.reset_game()  # reset AFTER preparing final_text
            return final_text
        
        #Max question limit activation
        if self.question_number >= 12:
            self.reset_game()
            #self.awaiting_user_animal = True
            return START_OVER_RESPONSE

        self.asked_features.add(feature)

        # Count questions
        self.question_number += 1

        self.last_feature_asked = feature

        use_extra = (self.question_number % 2 == 1)
        if feature in GuessingGameModel.global_asked_features:
            use_extra = False

        if use_extra and text_idx is not None:
            GuessingGameModel.global_asked_features.add(feature)
            return self.extra_text[text_idx] + response_text + f" {feature}?"
        else:
            return response_text + f" {feature}?"
        
    def validate_guess(self, user_answer):
        """
        Validate the user's yes/no answer for a guess.
        Returns (guessed_right: bool, text: str)
        """
        # Normalize answer
        ans = user_answer.lower().strip()

        if ans in ("yes", "y", "correct"):
            msg = f"Great! I guessed it right — you were thinking of {self.pending_guess_animal}!"
            self.pending_guess_animal = None
            return True, msg

        # When users says no, remove animal
        guessed = self.pending_guess_animal
        guessed_idx = self.y[self.y == guessed].index[0]

        # Remove from likelihood 
        self.likelihood = self.likelihood.drop(guessed_idx)

        msg = "Okay, let's keep going."
        self.pending_guess_animal = None
        return False, msg


    def get_next_feature(self):
        """Determine the next feature to ask about, handling stuck situations."""
        feature = None
        response_text = ""
        got_stuck = False
        text_idx = None
        
        if len(self.likelihood) > 1 \
            and check_stuck(self.top_history, self.likelihood, top_n=3, required_repeats=4) \
            and len(self.asked_features) > self.MIN_ASKED_FEATURES_BEFORE_STUCK_CHECK: 

            max_value = self.likelihood.max()
            top_candidates = self.likelihood[self.likelihood == max_value].index
            fallback = most_discriminative_feature(
                self.X, top_candidates, self.asked_features, self.useless_features
            )

            if fallback:
                response_text = STUCK_REASK_RESPONSE
                feature = fallback
                self.useless_features.add(feature)
            else:
                animals = list(self.y[top_candidates])[:3]
                response_text = STUCK_RESET_RESPONSE.format(stuck_animals=", ".join(animals))
                got_stuck = True
        else:
            top_candidates = self.likelihood[self.likelihood == self.likelihood.max()].index
            feature, text_idx = best_feature_to_ask(
                self.X, candidate_animals=top_candidates, asked_features=self.asked_features
            )

        return feature, response_text, got_stuck, text_idx


    def track_top_animals(self):
        top_vals = self.likelihood.sort_values(ascending=False)
        max_val = top_vals.iloc[0]
        tied = list(top_vals[top_vals == max_val].index)

        if len(tied) > 3:
            current_top = tuple(random.sample(tied, 3))
        else:
            current_top = tuple(top_vals.head(3).index)

        if len(self.asked_features) > 4:
            self.top_history.append(current_top)
        if len(self.top_history) > 5:
            self.top_history.pop(0)


    def reset_game(self):
        self.likelihood = pd.Series(0.0, index=self.X.index)
        self.asked_features = set()
        self.useless_features = set()
        self.turn_count = 0
        self.last_feature_asked = None
        self.pending_guess_animal = None
        self.top_history = []
