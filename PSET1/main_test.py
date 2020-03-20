'''CS236R PSET1, Florian Berlinger and Lily Xu, March 2020

A minimal version of predict.py running the highest performing model on the training data and performing predition on the test data.
'''

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

def write_to_csv(predictions):
    """Writes predictions to csv in 250x4 format (for evaluation against truth).

    Args:
        predictions (250x4 np-array of floats): predicted (f1, f2, f3, Action)
    """
    predictions = pd.DataFrame(predictions)
    predictions.columns = ['f1', 'f2', 'f3', 'action']
    predictions.to_csv('hb_test_pred.csv', index=False)

def reshape_feature(feature):
    """Reformat games

    Args:
        feature (18x1 np-array of ints): A single game

    Returns:
        tuple of 3x3 np-arrays of ints: Row and col player payoff matrices
    """
    row = feature[:9].reshape((3,3)) # payoff row player
    col = feature[9:].reshape((3,3)) # payoff column player

    return (row, col)

def nash_eq(features, subfunc=False):
    """Mixed-strategy Nash Equilibrium as per the Lemke Howson Algorithm

    Args:
        features (250x18 np-array of ints): Games
        subfunc (bool, optional): Won't print if used as subfunc of Hybrid

    Returns:
        predictions (250x4 np-array of floats): (f1, f2, f3, Action)
    """
    def pivot(A, r, s):
        """Helper function that pivots the tableau on the given row and column
        """

        m = len(A)
        B = A
        for i in range(m):
            if i == r:
                continue
            else:
                B[i,:] = A[i,:] - A[i,s] / A[r,s] * A[r,:]

        return B

    m = 3
    n = 3
    size_ = [m,n] # game size
    k0 = 0

    predictions = np.empty([len(features),4])
    feature_no = 0
    for feature in features:
        A, B = reshape_feature(feature)

        # positive payoffs only
        lowest = min(np.amin(A), np.amin(B))
        if lowest <= 0:
            A = A + (-lowest+1)
            B = B + (-lowest+1)

        # initialization of tableaux
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

    if not subfunc:
        Q_results.append(freq_dist)
        A_results.append(action_accuracy)
        descriptors.append('NASH EQUILIBRIUM')
        colors.append('r')
        print_to_terminal('NASH EQUILIBRIUM', freq_dist, action_accuracy)

    return predictions

# import data
train_features = pd.read_csv('hb_train_feature.csv')
train_features = train_features.to_numpy()
train_truths = pd.read_csv('hb_train_truth.csv')
train_truths = train_truths.to_numpy()
test_features = pd.read_csv('hb_test_feature.csv')
test_features = test_features.to_numpy()

# derive nash equilibria
train_nash_predictions = nash_eq(train_features, True)
train_features_neq = np.concatenate((train_features, train_nash_predictions), axis=1)

test_nash_predictions = nash_eq(test_features, True)
test_features_neq = np.concatenate((test_features, test_nash_predictions), axis=1)

# fit model on training data
SEED = 42 # random seed
regr = LinearRegression()
regr.fit(train_features_neq, train_truths)

# make model-based prediction on test data
predictions = regr.predict(test_features_neq)

# remove all negative frequencies
zero_idx = np.where(predictions < 0)[0]
if len(zero_idx) > 0:
    for i in range(len(zero_idx)):
        min_val = np.min(predictions[zero_idx[i], :3])
        predictions[zero_idx[i], :3] -= min_val

# normalize predictions and add action
freq_sum = predictions[:,:3].sum(axis=1)
predictions /= freq_sum[:, np.newaxis]
predictions[:,3] = np.argmax(predictions[:,:3], axis=1) + 1 # action

# write results to csv
write_to_csv(predictions)
