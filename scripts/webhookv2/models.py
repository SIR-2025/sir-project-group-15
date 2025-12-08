from guessing_algorithm import *
from utils import load_dataset
from responses import *

import pandas as pd
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
        self.last_feature_asked = None
        self.pending_guess_animal = None
        self.top_history = []
        
        self.MIN_ASKED_FEATURES_BEFORE_STUCK_CHECK = 5

        with open('extra_text.txt', 'r', encoding='utf-8') as f:
            self.extra_text = f.read().splitlines()

    def validate_guess(self, user_answer):
        if user_answer in ('yes', 'yep', 'yeah', 'y', 'correct'):
            return True, CORRECT_GUESS_RESPONSE.format(animal=self.pending_guess_animal)

        guessed_idx = self.y[self.y == self.pending_guess_animal].index[0]
        self.likelihood = self.likelihood.drop(guessed_idx)

        self.pending_guess_animal = None
        return False, WRONG_GUESS_RESPONSE


    def next_response(self, user_answer):
        """Main method to process user answer and return next question/guess."""
        
        response_text = ""

        # 1. If last turn was a guess, handle it
        if self.pending_guess_animal:
            guessed_right, response_text = self.validate_guess(user_answer)
            if guessed_right:
                return response_text
            # else continue to ask next question

        # 2. If we have a last feature, update likelihood
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

        # 3. Check if confident enough to guess
        if is_confident_enough(self.likelihood):
            best_idx = self.likelihood.idxmax()
            animal = self.y[best_idx]
            self.pending_guess_animal = animal
            return response_text + GUESS_RESPONSES.format(animal=animal)

        # 4. Otherwise pick the next feature
        if len(self.asked_features) == 0:
            feature = self.X.columns[self.first_feature_index]
            response_text = ""
            got_stuck = False
            text_idx = self.first_feature_index
        else:
            feature, response_text, got_stuck, text_idx = self.get_next_feature()


        if got_stuck:
            self.reset_game()
            return response_text

        # Mark the feature as asked
        self.asked_features.add(feature)

        # Save this feature so NEXT user_answer updates likelihood correctly
        self.last_feature_asked = feature

        # If this feature was asked in another game before → do NOT use extra_text
        if feature in GuessingGameModel.global_asked_features:
            return response_text + f" {feature}?"
        
        GuessingGameModel.global_asked_features.add(feature)
        if text_idx is None:
            return response_text + f" {feature}?"
        else:
            return self.extra_text[text_idx] + response_text + f" {feature}?"


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
