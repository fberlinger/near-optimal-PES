""" compare performance of different algorithms """

from lib_randomgraph import RandomGraph
from naive_flatrate import naive_flatrate
from all_combos import get_best_combination
from max_spanning_tree import spanning_tree #prim_MST
from greedy_node import greedy_node
from bellman_ford import select_bellman_ford
from connected_components import max_component #connected_components


import numpy as np
import matplotlib.pyplot as plt

budget = 20
graph = RandomGraph('Gnp', 10, edge_spec=.5, seed=42)

def get_values(combo):
    cost = graph.get_set_cost(combo)
    benefit = graph.get_set_benefit(combo)
    print('combination: {}'.format(combo))
    print('cost:    {}'.format(cost))
    print('benefit: {}'.format(benefit))
    print('value:   {}'.format(benefit - cost))
    return cost, benefit, benefit-cost

print(graph)

print('---------------------------')
print('best combination')
opt_combo, opt_value = get_best_combination(graph, budget)
opt_cost, opt_benefit, opt_value = get_values(opt_combo)

print('---------------------------')
print('naive flatrate')
flatrate = budget / graph.size
flatrate_combo = naive_flatrate(graph, budget, flatrate)
flatrate_cost, flatrate_benefit, flatrate_value = get_values(flatrate_combo)

print('---------------------------')
print('greedy node')
greedy_combo, greedy_value = greedy_node(graph, budget)
greedy_cost, greedy_benefit, greedy_value = get_values(greedy_combo)

print('---------------------------')
print('spanning tree')
spanning_combo, spanning_value = spanning_tree(graph, budget)
spanning_cost, spanning_benefit, spanning_value = get_values(spanning_combo)


print('---------------------------')
print('max connected component')
cc_combo = max_component(graph, budget)
cc_cost, cc_benefit, cc_value = get_values(cc_combo)


print('---------------------------')
print('bellman-ford')
bf_combo = select_bellman_ford(graph, budget)
bf_cost, bf_benefit, bf_value = get_values(bf_combo)


# display bar graph

display = [('best combo', opt_value),
           ('naive flatrate', flatrate_value),
           ('greedy node', greedy_value),
           ('MST', spanning_value),
           ('max CC', cc_value),
           ('bellman-ford', bf_value)]

plt.figure()
values = [entry[1] for entry in display]
X = np.arange(len(display))
plt.bar(X, values)
plt.xticks(X, [entry[0] for entry in display])
plt.ylabel('Value')
plt.xlabel('Method')
plt.show()
