
import numpy as np



def pivot(A, r, s):
# Pivots the tableau on the given row and column
    
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
size_ = [m,n]
k0 = 0

game = np.array([80, 80, 90, 90, 90, 50, 60, 80, 70, 80, 70, 60, 90, 90, 60, 30, 50, 20])

A = game[:9].reshape((m,n)) # payoff row player
B = game[9:].reshape((m,n)) # payoff column player
print(A)
print(B)

# (2) Initialization of Tableaux
tab = [[], []]
tab[0] = np.concatenate((np.transpose(B), np.eye(n), np.ones((n,1))), axis=1)
tab[1] = np.concatenate((np.eye(m), A, np.ones((m,1))), axis=1)

print(tab[0])
print(tab[1])

# row labels

row_labels = [[], []]
row_labels[0].extend(range(m,m+n))
row_labels[1].extend(range(0,m))

print(row_labels)

#
k = k0
if k0 <= m:
    player = 0
else:
    player = 1


# pivoting
while True:
    # Use correct Tableau
    LP = tab[player]
    print(LP)
    m_ = len(LP)
    
    # Find pivot row (variable exiting)
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
    
    # If the entering variable is the same
    # as the starting pivot, break
    if k == k0:
        break
    
    
    # update the tableau index
    if player == 0:
        player = 1
    else:
        player = 0
    

# Extract the Nash equilibrium
nash_eq = [[], []]

for player in range(2):
    print('player {}'.format(player))
    
    x = [0]*size_[player]
    rows = row_labels[player]
    LP = tab[player]

    for i in range(len(rows)):
        if player == 0 and rows[i] <= size_[1]-1:
            x[rows[i]] = LP[i][m+n] / LP[i][rows[i]]
            print(x)
        elif player == 1 and rows[i] > size_[1]-1:
            x[rows[i]-size_[1]] = LP[i][m+n] / LP[i][rows[i]]
            print(x)
    
    print(x)
    nash_eq[player] = x/sum(x)

print(nash_eq)
