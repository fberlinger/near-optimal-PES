
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

SEED = 42 # random seed


def write_to_csv(predictions):
    """Summary
    
    Args:
        predictions (TYPE): Description
    """
    predictions = pd.DataFrame(predictions)
    predictions.columns = ['f1', 'f2', 'f3', 'action']
    predictions.to_csv('hb_test_pred.csv', index=False)

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
regr = LinearRegression()
regr.fit(train_features_neq, train_truths)

# make model-based prediction on test data
predictions = regr.predict(test_features_neq)

# normalize predictions
freq_sum = predictions[:,:3].sum(axis=1)
predictions /= freq_sum[:, np.newaxis]
predictions[:,3] = np.argmax(predictions[:,:3], axis=1) + 1 # action

# write results to csv
write_to_csv(predictions)
