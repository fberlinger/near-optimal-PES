import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

games = pd.read_csv('hb_train_feature.csv')
games = games.to_numpy()
truths = pd.read_csv('hb_train_truth.csv')
truths = truths.to_numpy()


def eval_forecast(forecasts, truths):
    no_forecasts = len(forecasts)
    freq_dist = 0
    action_accuracy = 0

    for i in range(no_forecasts):
        freq_dist += (forecasts[i,0] - truths[i,0])**2 + (forecasts[i,1] - truths[i,1])**2 + (forecasts[i,2] - truths[i,2])**2

        action = np.argmax(forecasts[i,:3])
        #actions = max(forecasts[i,:3])
        #action = max(np.argwhere(forecasts[i,:3] == actions))
        if action == truths[i,3]-1: # 0 vs 1 index
            action_accuracy += 1

    freq_dist /= no_forecasts
    action_accuracy /= no_forecasts
    return (freq_dist, action_accuracy)

def random_guess():
    return np.random.rand(250,3)

def nash_eq(games):
    def pivot(A, r, s):
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

    forecasts = np.empty([len(games),3])
    game_no = 0
    for game in games:
        # data reformatting
        A = game[:9].reshape((3,3)) # payoff row player
        B = game[9:].reshape((3,3)) # payoff column player

        # initialization of Tableaux
        tab = [[], []]
        tab[0] = np.concatenate((np.transpose(B), np.eye(n), np.ones((n,1))), axis=1)
        tab[1] = np.concatenate((np.eye(m), A, np.ones((m,1))), axis=1)

        # row labels
        row_labels = [[], []]
        row_labels[0].extend(range(m,m+n))
        row_labels[1].extend(range(0,m))

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
            x = [0]*size_[player]
            rows = row_labels[player]
            LP = tab[player]

            for i in range(len(rows)):
                if player == 0 and rows[i] <= size_[1]-1:
                    x[rows[i]] = LP[i][m+n] / LP[i][rows[i]]
                elif player == 1 and rows[i] > size_[1]-1:
                    x[rows[i]-size_[1]] = LP[i][m+n] / LP[i][rows[i]]
                    
            nash_eq[player] = x/sum(x)

        forecasts[game_no,0] = nash_eq[0][0]
        forecasts[game_no,1] = nash_eq[0][1]
        forecasts[game_no,2] = nash_eq[0][2]
        game_no += 1

    return forecasts

def level_k(games, k):
    pass

def machine_learn():
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kfold.split(games):
        print('train', train_idx)
        print('test', test_idx)

#forecasts = random_guess()
forecasts = nash_eq(games)
#forecasts = level_k(games, 1)
freq_dist, action_accuracy = eval_forecast(forecasts, truths)
print(freq_dist, action_accuracy)
#machine_learn()
