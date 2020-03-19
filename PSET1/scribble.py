import numpy as np

a = np.array(([['f1', 'f2', 'f3', 'action']]))
b = np.array(([[1,2,3,4],[5,6,7,8]]))
print(a.shape)
print(b.shape)
c = np.concatenate((a,b))
print(c)

print(sign(-5))