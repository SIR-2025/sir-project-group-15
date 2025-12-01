def best_feature_to_ask(X, candidate_animals=None, asked_features=None):
    '''
    Function to determine the best possible question to ask. 
    Based on a split between the animals with the highest current likelihood
    '''
    if candidate_animals is None:
        X_sub = X
    else:
        X_sub = X.loc[candidate_animals]
    if asked_features:
        X_sub = X_sub.drop(columns=asked_features)
    best_feature = None
    best_split = 1.0
    for feature in X_sub.columns:
        yes_ratio = (X_sub[feature] == 1).mean()
        split_quality = abs(0.5 - yes_ratio)
        if split_quality < best_split:
            best_split = split_quality
            best_feature = feature
    return best_feature

def update_likelihood(animal_value, likelihood, i, answer):
    '''
    Function to update the likelihood values of all the animals after a question has been asked. 
    Add a value of 0/1 if a question was answered with certainty (yes/no). 
    Add a value of 0.25/0.75 if a question was answered as probably/probably not
    Add a value of 0.5 if a question was answered as i dont know
    '''
    if answer in ('yes', 'y'):
        likelihood.loc[i] += 1 if animal_value == 1 else 0
    elif answer in ('probably', 'probably yes'):
        likelihood.loc[i] += 0.75 if animal_value == 1 else 0.25
    if answer in ('i dont know', 'idk', 'maybe'):
        likelihood.loc[i] += 0.5
    if answer in ('probably not', 'probably no'):
        likelihood.loc[i] += 0.25 if animal_value == 1 else 0.75
    if answer in ('no', 'n'):
        likelihood.loc[i] += 0 if animal_value == 1 else 1
    return likelihood

def is_confident_enough(likelihood, margin=0.5):
    '''
    Method to check wether the animal with the highest likelihood value is significantly better than the rest.
    If so, return true, otherwise false
    '''
    sorted_vals = likelihood.sort_values(ascending=False)
    top = sorted_vals.iloc[0]
    second = sorted_vals.iloc[1]
    return (top - second >= margin)

def print_top_likelihoods(likelihood, y, top_n=5):
    '''
    Small unnecessary method to print the top 5 performing animals. Just for checking
    '''
    sorted_vals = likelihood.sort_values(ascending=False)
    
    print("\nTop likelihoods:")
    for idx, val in sorted_vals.head(top_n).items():
        print(f"- {y[idx]} : {val:.3f}")
    print()

def check_stuck(top_history, likelihood, top_n=3, required_repeats=4):
    '''
    Method to determine if the game gets stuck.
    It compares the top 3 performing animals. 
    If this top 3 remains the same for 4 consecutive questions this method will return true.
    '''
    current_top = tuple(likelihood.sort_values(ascending=False).head(top_n).index)
    if not top_history:
        return False
    if len(top_history) >= required_repeats - 1:
        if all(prev == current_top for prev in top_history[-(required_repeats):]):
            return True
    return False

def most_discriminative_feature(X, candidates, asked_features, useless_features):
    '''
    Method to determine after getting stuck which question to ask again. 
    If feature is in useless_feature, the feature has already been asked twice and cannot be asked again
    '''
    def best_feature_from_list(features):
        '''
        Intermediate method to determine which feature to ask from a small list of features based on the best performing animals.
        '''
        best_feature = None
        best_var = -1

        for f in features:
            if f in useless_features:
                continue
            values = X.loc[candidates, f]
            var = values.var()

            if var > best_var and var > 0:
                best_var = var
                best_feature = f

        return best_feature

    asked_list = [f for f in asked_features if f in X.columns and f not in useless_features]
    best = best_feature_from_list(asked_list)

    if best is not None:
        return best

    unasked = [f for f in X.columns if f not in asked_features]
    best = best_feature_from_list(unasked)

    if best is not None:
        return best

    return None
