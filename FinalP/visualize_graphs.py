""" code to visualize examples of random graph for figures """

import matplotlib.pyplot as plt
import networkx as nx

# gnp graph
n = 8
p = 0.3
gnp = nx.gnp_random_graph(n, p)
nx.draw_networkx(gnp)
plt.show()

# gnm graph
m = 16
gnm = nx.gnm_random_graph(n, m)
nx.draw_networkx(gnm)
plt.show()

# grid graph
grid = nx.grid_2d_graph(2, 4)
nx.draw(grid)
plt.show()
