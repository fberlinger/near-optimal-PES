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


def get_values(combo):
    cost = graph.get_set_cost(combo)
    benefit = graph.get_set_benefit(combo)
    print('combination: {}'.format(combo))
    print('cost:    {}'.format(cost))
    print('benefit: {}'.format(benefit))
    print('value:   {}'.format(benefit - cost))
    return cost, benefit, benefit-cost


num_repeats = 3
opt_values      = np.zeros(num_repeats)
flatrate_values = np.zeros(num_repeats)
greedy_values   = np.zeros(num_repeats)
spanning_values = np.zeros(num_repeats)
cc_values       = np.zeros(num_repeats)
bf_values       = np.zeros(num_repeats)

for i in range(num_repeats):
    budget = 10
    graph = RandomGraph('Gnp', 4, edge_spec=.5, seed=None)
    print(graph)


    print('---------------------------')
    print('best combination')
    opt_combo, opt_value = get_best_combination(graph, budget)
    opt_cost, opt_benefit, opt_values[i] = get_values(opt_combo)

    print('---------------------------')
    print('naive flatrate')
    flatrate = budget / graph.size
    flatrate_combo = naive_flatrate(graph, budget, flatrate)
    flatrate_cost, flatrate_benefit, flatrate_values[i] = get_values(flatrate_combo)

    print('---------------------------')
    print('greedy node')
    greedy_combo, greedy_value = greedy_node(graph, budget)
    greedy_cost, greedy_benefit, greedy_values[i] = get_values(greedy_combo)

    # print('---------------------------')
    # print('spanning tree')
    # spanning_combo, spanning_value = spanning_tree(graph, budget)
    # spanning_cost, spanning_benefit, spanning_values[i] = get_values(spanning_combo)


    print('---------------------------')
    print('max connected component')
    cc_combo = max_component(graph, budget)
    cc_cost, cc_benefit, cc_values[i] = get_values(cc_combo)


    print('---------------------------')
    print('bellman-ford')
    bf_combo = select_bellman_ford(graph, budget)
    bf_cost, bf_benefit, bf_values[i] = get_values(bf_combo)


# display bar graph
display = [('best combo', opt_values),
           ('naive flatrate', flatrate_values),
           ('greedy node', greedy_values),
           # ('MST', spanning_values),
           ('max CC', cc_values),
           ('bellman-ford', bf_values)]

plt.figure()
values = [entry[1].mean() for entry in display]
X = np.arange(len(display))
plt.bar(X, values, yerr=[entry[1].std() for entry in display])
plt.xticks(X, [entry[0] for entry in display])
plt.ylabel('Value')
plt.xlabel('Method')
plt.show()
