"""CS236R PSET1, Florian Berlinger and Lily Xu, March 2020 

Attributes:
    A_results (list): Description
    behaviors (list): Description
    colors (list): Description
    descriptors (list): Description
    models (list): Description
    predictions (TYPE): Description
    Q_results (list): Description
    SEED (int): Description
    test_features (TYPE): Description
    train_features (TYPE): Description
    train_truths (TYPE): Description
    training (bool): Description

Deleted Attributes:
    action_accuracy (TYPE): Description
    avg_action_accuracy (TYPE): Description
    avg_freq_dist (TYPE): Description
    freq_dist (TYPE): Description
    num_repeats (int): Description
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


def eval_forecast(predictions, truths):
    """return (Q, A) where
    Q = quadratic distance of the frequency distribution
    A = accuracy 
    
    Args:
        predictions (TYPE): Description
        truths (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    no_predictions = len(predictions)
    freq_dist = 0
    action_accuracy = 0

    for ii in range(no_predictions):
        freq_dist += (predictions[ii, 0] - truths[ii, 0])**2 + (predictions[ii, 1] - truths[ii, 1])**2 + (predictions[ii, 2] - truths[ii, 2])**2

        action = predictions[ii, 3]
        if action == truths[ii, 3]:
            action_accuracy += 1

    freq_dist /= no_predictions
    action_accuracy /= no_predictions

    return (freq_dist, action_accuracy)

def print_to_terminal(exp, Q, A):
    """Summary
    
    Args:
        exp (TYPE): Description
        Q (TYPE): Description
        A (TYPE): Description
    """
    print(exp)
    print('Q = {:.3f}, A = {:.3f}\n'.format(Q, A))

def plot(Q, A, descriptors, winner, colors):
    """Summary
    
    Args:
        Q (TYPE): Description
        A (TYPE): Description
        descriptors (TYPE): Description
        colors (TYPE): Description
        winner (TYPE): Description
    """
    
    fig, axs = plt.subplots(figsize=(10,6))
    axs.scatter(Q, A, s=120, c=colors, alpha=0.5)
    axs.set_xlabel('Q', fontsize='large', fontweight='bold')
    axs.set_ylabel('A', fontsize='large', fontweight='bold')
    axs.grid('k')
    axs.annotate(descriptors[winner], (Q[winner], A[winner]))
    axs.text(0.8, 0.9, 'red: behavior\nblue: ML\npurple: hybrid', ha='left', va='top', transform=axs.transAxes, fontsize=10)
    plt.tight_layout()
    plt.savefig('Q&A.png')
    plt.close()

def write_to_csv(predictions):
    """Summary
    
    Args:
        predictions (TYPE): Description
    """
    predictions = pd.DataFrame(predictions)
    predictions.columns = ['f1', 'f2', 'f3', 'action']
    predictions.to_csv('hb_test_pred.csv', index=False)

def random_guess(num_repeats, subfunc=False):
    """baseline: random guessing 
    
    Returns:
        numpy array of floats (len(features), 4): normalized random freqs and action
    
    Args:
        num_repeats (TYPE): Description
        subfunc (bool, optional): Description
    """
    freq_dist = np.zeros(num_repeats)
    action_accuracy = np.zeros(num_repeats)

    for i in range(num_repeats):
        predictions = np.empty([len(features), 4])
        predictions[:, :3] = np.random.rand(250, 3) # random freqs
        freq_sum = predictions[:, :3].sum(axis=1) # normalize
        predictions[:, :3] /= freq_sum[:, np.newaxis]
        predictions[:, 3] = np.argmax(predictions[:, :3], axis=1) + 1 # action

        freq_dist[i], action_accuracy[i] = eval_forecast(predictions, truths)

    avg_freq_dist = np.mean(freq_dist)
    avg_action_accuracy = np.mean(action_accuracy)

    if not subfunc:
        Q_results.append(avg_freq_dist)
        A_results.append(avg_action_accuracy)
        descriptors.append('RANDOM GUESSING')
        colors.append('r')
        print_to_terminal('RANDOM GUESSING', avg_freq_dist, avg_action_accuracy)

        return predictions # last one
    
    else:
        return (predictions, avg_freq_dist, avg_action_accuracy)

def reshape_feature(feature):
    """reformat data 
    
    Args:
        feature (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    row = feature[:9].reshape((3,3)) # payoff row player
    col = feature[9:].reshape((3,3)) # payoff column player

    return row, col

def nash_eq(features, subfunc=False):
    """mixed-strategy Nash equilibrium 
    
    Args:
        features (TYPE): Description
        subfunc (bool, optional): Description
    
    Returns:
        TYPE: Description
    """
    def pivot(A, r, s):
        """Summary
        
        Args:
            A (TYPE): Description
            r (TYPE): Description
            s (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # pivots the tableau on the given row and column
        m = len(A)
        B = A
        for i in range(m):
            if i == r:
                continue
            else:
                B[i,:] = A[i,:] - A[i,s] / A[r,s] * A[r,:]

        return B

    # Lemke Howson algorithm
    m = 3
    n = 3
    size_ = [m,n]
    k0 = 0

    predictions = np.empty([len(features),4])
    feature_no = 0
    for feature in features:
        A, B = reshape_feature(feature)

        lowest = min(np.amin(A), np.amin(B))
        if lowest <= 0:
            A = A + (-lowest+1)
            B = B + (-lowest+1)

        # initialization of Tableaux
        tab = [[], []]
        tab[0] = np.concatenate((np.transpose(B), np.eye(n), np.ones((n,1))), axis=1)
        tab[1] = np.concatenate((np.eye(m), A, np.ones((m,1))), axis=1)

        # row labels
        row_labels = [[], []]
        row_labels[0].extend(range(m, m+n))
        row_labels[1].extend(range(0, m))

        # initial player
        k = k0
        if k0 <= m:
            player = 0
        else:
            player = 1

        # pivoting
        while True:
            # choose tableau
            LP = tab[player]
            m_ = len(LP)

            # find pivot row (variable exiting)
            max_ = 0
            ind = -1
            for i in range(m_):
                if LP[i, m+n] == 0:
                    if LP[i, k] > 0:
                        t = math.inf
                    else:
                        t = -math.inf
                else:
                    t = LP[i, k] / LP[i, m+n]
                if t > max_:
                    ind = i
                    max_ = t

            if max_ > 0:
                tab[player] = pivot(LP, ind, k)
            else:
                break

            # swap labels, set entering variable
            temp = row_labels[player][ind]
            row_labels[player][ind] = k
            k = temp

            # if the entering variable is the same as the starting pivot, break
            if k == k0:
                break

            # update the tableau index
            if player == 0:
                player = 1
            else:
                player = 0

        # extract the Nash equilibrium
        nash_eq = [[], []]

        for player in range(2):
            x = [0] * size_[player]
            rows = row_labels[player]
            LP = tab[player]

            for i in range(len(rows)):
                if player == 0 and rows[i] <= size_[1]-1:
                    x[rows[i]] = LP[i][m+n] / LP[i][rows[i]]
                elif player == 1 and rows[i] > size_[1]-1:
                    x[rows[i]-size_[1]] = LP[i][m+n] / LP[i][rows[i]]

            nash_eq[player] = x/sum(x)

        predictions[feature_no,0] = nash_eq[0][0] # f1
        predictions[feature_no,1] = nash_eq[0][1] # f2
        predictions[feature_no,2] = nash_eq[0][2] # f3
        predictions[feature_no,3] = np.argmax(predictions[feature_no,:3]) + 1 # action (1-indexed)
        feature_no += 1

    (freq_dist, action_accuracy) = eval_forecast(predictions, truths)
    if not subfunc:
        Q_results.append(freq_dist)
        A_results.append(action_accuracy)
        descriptors.append('NASH EQUILIBRIUM')
        colors.append('r')

        print_to_terminal('NASH EQUILIBRIUM', freq_dist, action_accuracy)

    return predictions

def level_k(features, k, subfunc=False):
    """level-k behavioral model 
    
    Args:
        features (TYPE): Description
        k (TYPE): Description
        subfunc (bool, optional): Description
    
    Returns:
        TYPE: Description
    
    Raises:
        Exception: Description
    """
    def max_action(payoff):
        """given a 3x3 payoff matrix, calculate max payoff
        assuming opponent plays max action 
        
        Args:
            payoff (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        avg_payoff = payoff.mean(axis=1)
        action = np.argmax(avg_payoff)
        return action

    if k == 0:
        # level-0: random strategy
        (predictions, freq_dist, action_accuracy) = random_guess(1000, True)
    elif k == 1:
        # level-1: max payoff assuming column player uses random strategy
        predictions = np.zeros((len(features), 4))
        for i, feature in enumerate(features):
            row, col = reshape_feature(feature)
            action = max_action(row)
            predictions[i, action] = 1
            predictions[i, 3] = action + 1
    elif k == 2:
        # level-2: max payoff assuming column player uses level-1
        predictions = np.zeros((len(features), 4))
        for i, feature in enumerate(features):
            row, col = reshape_feature(feature)
            col_action = max_action(col.T)

            action = np.argmax(row[:, col_action])
            predictions[i, action] = 1
            predictions[i, 3] = action + 1
    else:
        raise Exception('level-k not implemented for k={}'.format(k))    

    if not subfunc:
        if not k == 0:
            (freq_dist, action_accuracy) = eval_forecast(predictions, truths)
        Q_results.append(freq_dist)
        A_results.append(action_accuracy)
        descriptors.append('LEVEL-{}'.format(k))
        colors.append('r')

        print_to_terminal('LEVEL-{}'.format(k), freq_dist, action_accuracy)

    return predictions

def ml_predict(X, y, X_test, model):
    """return predictionsictions using specified ML model
    helper function used by machine_learn and hybrid 
    
    Args:
        X (TYPE): Description
        y (TYPE): Description
        X_test (TYPE): Description
        model (TYPE): Description
    
    Returns:
        TYPE: Description
    
    Raises:
        Exception: Description
    """
    if model == 'Random Forest':
        regr = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=SEED)
    elif model == 'Linear Regression':
        regr = LinearRegression()
    elif model == 'Decision Tree':
        regr = DecisionTreeRegressor()
    elif model == 'Gradient Boost':
        single_regr = GradientBoostingRegressor(n_estimators=100, random_state=SEED)
        regr = MultiOutputRegressor(single_regr)
    elif model == 'Ada Boost':
        single_regr = AdaBoostRegressor(n_estimators=100, random_state=SEED)
        regr = MultiOutputRegressor(single_regr)
    else:
        raise Exception('model {} not recognized'.format(model))

    regr.fit(X, y)

    y_predict = regr.predict(X_test)

    # normalize predictions
    freq_sum = y_predict[:,:3].sum(axis=1)
    y_predict /= freq_sum[:, np.newaxis]


    y_predict[:,3] = np.argmax(y_predict[:,:3], axis=1) + 1 # action
    return y_predict

def machine_learn(features, model, n_splits=5):
    """pure machine learning 
    
    Args:
        model (TYPE): Description
        n_splits (int, optional): Description
    
    Returns:
        TYPE: Description
    """
    freq_dist = np.zeros(n_splits)
    action_accuracy = np.zeros(n_splits)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for i, (train_idx, test_idx) in enumerate(kfold.split(features)):
        X = features[train_idx]
        y = truths[train_idx]

        X_test = features[test_idx]
        y_test = truths[test_idx]

        y_predict = ml_predict(X, y, X_test, model)
        freq_dist[i], action_accuracy[i] = eval_forecast(y_predict, y_test)

    avg_freq_dist = np.mean(freq_dist)
    avg_action_accuracy = np.mean(action_accuracy)
    
    Q_results.append(avg_freq_dist)
    A_results.append(avg_action_accuracy)
    descriptors.append('ML {}'.format(model))
    colors.append('b')
    
    print_to_terminal('ML {}'.format(model), avg_freq_dist, avg_action_accuracy)

    return y_predict

def hybrid(features, behavior, model, n_splits=5):
    """add behavior model predictionsictions as ML features 
    
    Args:
        behavior (TYPE): Description
        model (TYPE): Description
        n_splits (int, optional): Description
    
    Returns:
        TYPE: Description
    
    Raises:
        Exception: Description
    """
    if behavior == 'Nash Equilibrium':
        predictions = nash_eq(features, True)
    elif behavior == 'Level-0':
        predictions = level_k(features, 0, True)
    elif behavior == 'Level-1':
        predictions = level_k(features, 1, True)
    elif behavior == 'Level-2':
        predictions = level_k(features, 2, True)
    else:
        raise Exception('behavior {} not recognized'.format(behavior))


    all_X = np.concatenate((features, predictions), axis=1)

    freq_dist = np.zeros(n_splits)
    action_accuracy = np.zeros(n_splits)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for i, (train_idx, test_idx) in enumerate(kfold.split(features)):
        X = all_X[train_idx]
        y = truths[train_idx]

        X_test = all_X[test_idx]
        y_test = truths[test_idx]

        y_predict = ml_predict(X, y, X_test, model)
        freq_dist[i], action_accuracy[i] = eval_forecast(y_predict, y_test)

    avg_freq_dist = np.mean(freq_dist)
    avg_action_accuracy = np.mean(action_accuracy)

    Q_results.append(avg_freq_dist)
    A_results.append(avg_action_accuracy)
    descriptors.append('HYBRID {} + {}'.format(behavior, model))
    colors.append('purple')

    print_to_terminal('HYBRID {} + {}'.format(behavior, model), avg_freq_dist, avg_action_accuracy)

    return y_predict


if __name__ == "__main__":
    SEED = 42  # random seed

    # import data
    train_features = pd.read_csv('hb_train_feature.csv')
    train_features = train_features.to_numpy()
    train_truths = pd.read_csv('hb_train_truth.csv')
    train_truths = train_truths.to_numpy()
    test_features = pd.read_csv('hb_test_feature.csv')
    test_features = test_features.to_numpy()

    training = True
    if training:
        features = train_features
        truths = train_truths

    models = ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boost', 'Ada Boost']

    Q_results = []
    A_results = []
    descriptors = []
    colors = []

    ## RANDOM GUESSING
    predictions = random_guess(1000)

    ## NASH EQUILIBRIUM
    predictions = nash_eq(features)

    ## LEVEL-K
    predictions = level_k(features, 0)
    predictions = level_k(features, 1)
    predictions = level_k(features, 2)

    ## PURE ML
    for model in models:
        predictions = machine_learn(features, model)

    ## ML HYBRIDS
    behaviors = ['Nash Equilibrium', 'Level-0', 'Level-1', 'Level-2']
    for behavior in behaviors:
        for model in models:
            predictions = hybrid(features, behavior, model)

    winner = np.argmax(A_results)
    print('The winning model is: {} with Q = {:.3f} and A = {:.3f}.'.format(descriptors[winner], Q_results[winner], A_results[winner]))
    plot(Q_results, A_results, descriptors, winner, colors)
