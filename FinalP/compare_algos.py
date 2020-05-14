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


def get_values(graph, budget, combo, verbose=False):
    graph.check_set_valid(combo, budget)
    cost    = graph.get_set_cost(combo)
    benefit = graph.get_set_benefit(combo)
    if verbose:
        print('combination: {}'.format(combo))
        print('cost:    {}'.format(cost))
        print('benefit: {}'.format(benefit))
        print('value:   {}'.format(benefit - cost))
    return cost, benefit, benefit-cost


def bar_values():
    """ run experiments and plot bar graph showing value attained by each algorithm """

    num_repeats = 5
    budget      = 30
    num_nodes   = 15
    edge_spec   = .5

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
        graph = RandomGraph('Gnp', num_nodes, edge_spec=edge_spec, seed=None)
        print('\n\n', graph)

        print('---------------------------')
        print('best combination')
        opt_combo, opt_value = get_best_combination(graph, budget)
        opt_costs[i], opt_benefits[i], opt_values[i] = get_values(graph, budget, opt_combo)

        print('---------------------------')
        print('naive flat rate')
        # flatrate = budget / graph.size * 8
        flatrate = 6
        flatrate_combo = naive_flatrate(graph, budget, flatrate)
        flatrate_costs[i], flatrate_benefits[i], flatrate_values[i] = get_values(graph, budget, flatrate_combo)

        print('---------------------------')
        print('greedy node')
        greedy_combo, greedy_value = greedy_node(graph, budget)
        greedy_costs[i], greedy_benefits[i], greedy_values[i] = get_values(graph, budget, greedy_combo)

        print('---------------------------')
        print('spanning tree')
        spanning_combo, spanning_value = spanning_tree(graph, budget)
        spanning_costs[i], spanning_benefits[i], spanning_values[i] = get_values(graph, budget, spanning_combo)

        print('---------------------------')
        print('max connected component')
        cc_combo = max_component(graph, budget)
        cc_costs[i], cc_benefits[i], cc_values[i] = get_values(graph, budget, cc_combo)

        print('---------------------------')
        print('bellman-ford')
        bf_combo = select_bellman_ford(graph, budget)
        bf_costs[i], bf_benefits[i], bf_values[i] = get_values(graph, budget, bf_combo)


    # display bar graph
    display = [('optimal', opt_costs.mean(), opt_benefits.mean(), opt_values.mean()), # no error bar for best
               ('naive flat rate', flatrate_costs, flatrate_benefits, flatrate_values),
               ('greedy node', greedy_costs, greedy_benefits, greedy_values),
               ('MST', spanning_costs, spanning_benefits, spanning_values),
               ('max CC', cc_costs, cc_benefits, cc_values),
               ('bellman-ford', bf_costs, bf_benefits, bf_values)]


    # import sys
    # sys.exit(0)
    # plt.figure()
    # values = [entry[1].mean() for entry in display]
    # X = np.arange(len(display))
    # plt.bar(X, values, yerr=[entry[1].std() for entry in display], color='red')
    # plt.xticks(X, [entry[0] for entry in display], rotation=30)
    # plt.ylabel('Cost')
    # plt.xlabel('Method')
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure()
    # values = [entry[2].mean() for entry in display]
    # X = np.arange(len(display))
    # plt.bar(X, values, yerr=[entry[2].std() for entry in display], color='green')
    # plt.xticks(X, [entry[0] for entry in display], rotation=30)
    # plt.ylabel('Benefit')
    # plt.xlabel('Method')
    # plt.tight_layout()
    # plt.show()

    # normalize by percentage of optimal achieved
    values = [100 * (entry[3] / display[0][3]).mean() for entry in display]
    stdev =  [100 * (entry[3] / display[0][3]).std() for entry in display]

    plt.figure()
    X = np.arange(len(display))
    plt.bar(X, values, yerr=stdev, color='Blue')
    plt.xticks(X, [entry[0] for entry in display], rotation=30)
    plt.ylabel('Value (% of optimal)')
    plt.xlabel('Method')
    plt.tight_layout()
    plt.show()
    plt.savefig('plot_bar_value.png')

def vary_budget():
    budgets     = [5, 20, 25, 30, 40, 50, 70, 100]
    num_nodes   = 15
    edge_spec   = .3
    num_repeats = 30

    opt_values      = np.zeros((len(budgets), num_repeats))
    flatrate_values = np.zeros((len(budgets), num_repeats))
    greedy_values   = np.zeros((len(budgets), num_repeats))
    spanning_values = np.zeros((len(budgets), num_repeats))
    cc_values       = np.zeros((len(budgets), num_repeats))
    bf_values       = np.zeros((len(budgets), num_repeats))

    np.random.seed(42)
    seeds = np.random.random(num_repeats)

    for i in range(num_repeats):
        graph = RandomGraph('Gnp', num_nodes, edge_spec=edge_spec, seed=seeds[i])
        print('\n\n', graph)
        for b, budget in enumerate(budgets):

            # print('---------------------------')
            # print('best combination')
            opt_combo, opt_value = get_best_combination(graph, budget)
            _, _, opt_values[b][i] = get_values(graph, budget, opt_combo)

            # print('---------------------------')
            # print('naive flat rate')
            # flatrate = budget / graph.size * 8
            flatrate = 4
            flatrate_combo = naive_flatrate(graph, budget, flatrate)
            _, _, flatrate_values[b][i] = get_values(graph, budget, flatrate_combo)

            # print('---------------------------')
            # print('greedy node')
            greedy_combo, greedy_value = greedy_node(graph, budget)
            _, _, greedy_values[b][i] = get_values(graph, budget, greedy_combo)

            # print('---------------------------')
            # print('spanning tree')
            spanning_combo, spanning_value = spanning_tree(graph, budget)
            _, _, spanning_values[b][i] = get_values(graph, budget, spanning_combo)

            # print('---------------------------')
            # print('max connected component')
            cc_combo = max_component(graph, budget)
            _, _, cc_values[b][i] = get_values(graph, budget, cc_combo)

            # print('---------------------------')
            # print('bellman-ford')
            bf_combo = select_bellman_ford(graph, budget)
            _, _, bf_values[b][i] = get_values(graph, budget, bf_combo)

    # print(opt_values)
    # print(flatrate_values)
    # print(greedy_values)
    # print(spanning_values)
    # print(cc_values)
    # print(bf_values)

    opt_values_norm = 100 * (opt_values / opt_values)
    flatrate_values = 100 * (flatrate_values / opt_values)
    greedy_values   = 100 * (greedy_values / opt_values)
    spanning_values = 100 * (spanning_values / opt_values)
    cc_values       = 100 * (cc_values / opt_values)
    bf_values       = 100 * (bf_values / opt_values)


    plt.figure()
    plt.errorbar(budgets, opt_values_norm.mean(axis=1), yerr=None,
                          label='optimal', #ecolor='black',
                          elinewidth=.5, color='forestgreen')
    plt.errorbar(budgets, flatrate_values.mean(axis=1), yerr=flatrate_values.std(axis=1),
                          label='naive flat rate', #ecolor='black',
                          elinewidth=.5, color='gold')
    plt.errorbar(budgets, greedy_values.mean(axis=1), yerr=greedy_values.std(axis=1),
                          label='greedy node', #ecolor='black',
                          elinewidth=.5, color='red')
    plt.errorbar(budgets, spanning_values.mean(axis=1), yerr=spanning_values.std(axis=1),
                          label='MST', #ecolor='black',
                          elinewidth=.5, color='royalblue')
    plt.errorbar(budgets, cc_values.mean(axis=1), yerr=cc_values.std(axis=1),
                          label='max CC', #ecolor='black',
                          elinewidth=.5, color='darkturquoise')
    plt.errorbar(budgets, bf_values.mean(axis=1), yerr=bf_values.std(axis=1),
                          label='bellman-ford', #ecolor='black',
                          elinewidth=.5, color='darkviolet')
    plt.legend()
    plt.xlabel('Budget')
    plt.ylabel('Average value (% of optimal)')
    plt.tight_layout()
    plt.savefig('plot_vary_budget.png')
    plt.show()

    # # plot optimal value only
    # plt.figure()
    # plt.plot(budgets, opt_values.mean(axis=1), label='optimal', color='forestgreen', marker='o', linestyle='-')
    # plt.legend()
    # plt.xlabel('Budget')
    # plt.ylabel('Value')
    # plt.tight_layout()
    # plt.savefig('plot_vary_budget_optimal.png')
    # plt.show()



if __name__ == '__main__':
    # bar_values()
    vary_budget()
