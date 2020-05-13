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


num_repeats = 1
opt_costs      = np.zeros(num_repeats)
flatrate_costs = np.zeros(num_repeats)
greedy_costs   = np.zeros(num_repeats)
spanning_costs = np.zeros(num_repeats)
cc_costs       = np.zeros(num_repeats)
bf_costs       = np.zeros(num_repeats)

opt_benefits      = np.zeros(num_repeats)
flatrate_benefits = np.zeros(num_repeats)
greedy_benefits   = np.zeros(num_repeats)
spanning_benefits = np.zeros(num_repeats)
cc_benefits       = np.zeros(num_repeats)
bf_benefits       = np.zeros(num_repeats)

opt_values      = np.zeros(num_repeats)
flatrate_values = np.zeros(num_repeats)
greedy_values   = np.zeros(num_repeats)
spanning_values = np.zeros(num_repeats)
cc_values       = np.zeros(num_repeats)
bf_values       = np.zeros(num_repeats)

for i in range(num_repeats):
    budget = 20
    graph = RandomGraph('Gnp', 15, edge_spec=.5, seed=0)
    print('\n\n', graph)

    print('---------------------------')
    print('best combination')
    opt_combo, opt_value = get_best_combination(graph, budget)
    opt_costs[i], opt_benefits[i], opt_values[i] = get_values(opt_combo)

    print('---------------------------')
    print('naive flatrate')
    flatrate = budget / graph.size
    flatrate_combo = naive_flatrate(graph, budget, flatrate)
    flatrate_costs[i], flatrate_benefits[i], flatrate_values[i] = get_values(flatrate_combo)

    print('---------------------------')
    print('greedy node')
    greedy_combo, greedy_value = greedy_node(graph, budget)
    greedy_costs[i], greedy_benefits[i], greedy_values[i] = get_values(greedy_combo)

    print('---------------------------')
    print('spanning tree')
    spanning_combo, spanning_value = spanning_tree(graph, budget)
    spanning_costs[i], spanning_benefits[i], spanning_values[i] = get_values(spanning_combo)

    print('---------------------------')
    print('max connected component')
    cc_combo = max_component(graph, budget)
    cc_costs[i], cc_benefits[i], cc_values[i] = get_values(cc_combo)

    print('---------------------------')
    print('bellman-ford')
    bf_combo = select_bellman_ford(graph, budget)
    bf_costs[i], bf_benefits[i], bf_values[i] = get_values(bf_combo)


# display bar graph
display = [('best combo', opt_costs.mean(), opt_benefits.mean(), opt_values.mean()), # no error bar for best
           ('naive flatrate', flatrate_costs, flatrate_benefits, flatrate_values),
           ('greedy node', greedy_costs, greedy_benefits, greedy_values),
           ('MST', spanning_costs, spanning_benefits, spanning_values),
           ('max CC', cc_costs, cc_benefits, cc_values),
           ('bellman-ford', bf_costs, bf_benefits, bf_values)]


# import sys
# sys.exit(0)
plt.figure()
values = [entry[1].mean() for entry in display]
X = np.arange(len(display))
plt.bar(X, values, yerr=[entry[1].std() for entry in display], color='red')
plt.xticks(X, [entry[0] for entry in display], rotation=30)
plt.ylabel('Cost')
plt.xlabel('Method')
plt.tight_layout()
plt.show()

plt.figure()
values = [entry[2].mean() for entry in display]
X = np.arange(len(display))
plt.bar(X, values, yerr=[entry[2].std() for entry in display], color='green')
plt.xticks(X, [entry[0] for entry in display], rotation=30)
plt.ylabel('Benefit')
plt.xlabel('Method')
plt.tight_layout()
plt.show()

plt.figure()
values = [entry[3].mean() for entry in display]
X = np.arange(len(display))
plt.bar(X, values, yerr=[entry[3].std() for entry in display], color='Blue')
plt.xticks(X, [entry[0] for entry in display], rotation=30)
plt.ylabel('Value')
plt.xlabel('Method')
plt.tight_layout()
plt.show()
