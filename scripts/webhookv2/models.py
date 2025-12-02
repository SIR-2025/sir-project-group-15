from guessing_algorithm import *
from utils import load_dataset

import pandas as pd
import random

class GuessingGameModel:
    def __init__(self, session_id, animal_dataset_path):
        self.session_id = session_id
        self.X, self.y = load_dataset(animal_dataset_path)
        
        self.likelihood = pd.Series(0.0, index=self.X.index)
        self.asked_features = set()
        self.useless_features = set()
        self.turn_count = 0
        self.last_feature_asked = None
        self.pending_guess_animal = None
        self.top_history = []
        
        self.MIN_ASKED_FEATURES_BEFORE_STUCK_CHECK = 5
        

    def validate_guess(self, user_answer):
        response_text = ""
        
        if user_answer in ('yes', 'yep', 'yeah', 'y', 'correct'):
            response_text = "yipie, I got it correct! Say 'reset' to play again."
        else:
            # Find the index of the animal we just guessed and remove it.
            guessed_idx = self.y[self.y == self.pending_guess_animal].index[0]
            self.likelihood = self.likelihood.drop(guessed_idx)
            
            response_text = "Okay, not that. Let me think... "
            self.pending_guess_animal = None 
            
        return response_text
            
            
    def next_response(self, user_answer):
        '''
        If statement to check if highest likelihood animal can be guessed.
        When wrong animal, this animal will be removed from the possibilities.
        '''    
        response_text = ""
        
        # if not (self.last_feature_asked and user_answer):
        #     return response_text
        
        if self.pending_guess_animal:
            guessed_right, response_text = self.validate_guess(user_answer)
            if guessed_right:
                return response_text
        
        if is_confident_enough(self.likelihood): 
            best_idx = self.likelihood.idxmax()
            best_animal = self.y[best_idx]
            response_text = f"Is it a {best_animal}?"
            self.pending_guess_animal = best_animal
        else:   
            feature, response_text, got_stuck = self.get_next_feature()
            if got_stuck:
                self.reset_game()
                return response_text
                
            
            self.asked_features.add(feature)
            response_text += f" {feature}?"
            
            for i in self.likelihood.index: # Update likelihood for next round
                self.likelihood = update_likelihood(self.X.loc[i, feature], self.likelihood, i, user_answer)
            
            self.track_top_animals()
        
        return response_text
    
    
    def get_next_feature(self):
        '''
        Part to determine which feature to extract next to ask. If check_stuck method is activated, the algorithm is not updating the top candidates
        anymore and therefore will ask a question previously answered with "i dont know" again. 
        Else, the algorithm will just ask a feature based on the optimal split for the top performing animals.
        '''
        feature = None
        response_text = ""
        got_stuck = False
        
        if len(self.likelihood) > 1 \
            and check_stuck(self.top_history, self.likelihood, top_n=3, required_repeats=4) \
            and len(self.asked_features) > self.MIN_ASKED_FEATURES_BEFORE_STUCK_CHECK: 
                
            max_value = self.likelihood.max()
            top_candidates = self.likelihood[self.likelihood == max_value].index
            fallback_feature = most_discriminative_feature(self.X, top_candidates, self.asked_features, self.useless_features)
            
            if fallback_feature:
                response_text = "It seems I'm not getting new information from your answers. Please let me ask you this specific question again:"
                feature = fallback_feature
                self.useless_features.add(feature)
            else:
                animal_names = list(self.y[top_candidates])
                animal_names = animal_names[:3] 
                response_text = f"I seem to be stuck between the following animals: {', '.join(animal_names)}. I will try to guess it next time."
                got_stuck = True
        else:
            top_candidates = self.likelihood[self.likelihood == self.likelihood.max()].index
            feature = best_feature_to_ask(self.X, candidate_animals=top_candidates, asked_features=self.asked_features)
            
        return feature, response_text, got_stuck

    def track_top_animals(self):
        '''
        This code keeps track of the top performing animals which will be fed into check_stuck.
        If the top performing animals are exactly the same after a few questions, the model gets stuck and has to ask an already asked question again. 
        '''
        current_top = tuple(self.likelihood.sort_values(ascending=False).head(3).index)
        top_vals = self.likelihood.sort_values(ascending=False)
        max_val = top_vals.iloc[0]
        tied = list(top_vals[top_vals == max_val].index)
        
        if len(tied) > 3:
            current_top = tuple(random.sample(tied, 3))
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
        self.pending_guess_animal = False
        self.top_history = []
        