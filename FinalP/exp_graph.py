"""Experiments on algorithms that find good PES solutions for graph representations of the problem.
"""
import math
import random
import time
import sys

from lib_randomgraph import RandomGraph
from lib_heap import Heap


#### RANDOM GRAPH #############################################################

graph_type = 'Gnm'
graph_size = 20 # n
no_edges = 10 # m
G = RandomGraph(graph_type, graph_size, no_edges)
#print(G.graph)
#print(G.node_weights)
exp_graph = G.graph
exp_node_weights = G.node_weights
no_nodes = graph_size

budget = graph_size*1.5 # example budget

###############################################################################



#### SIMPLE EXAMPLE GRAPH #####################################################

# number of nodes
no_nodes = 4
# adjacency list
# format: [neighbor, edge weight]
exp_graph = [[[1, 4], [2, 7]], [[2, 1], [0, 4], [3, 9]], [[1, 1], [0, 7]], [[1, 9]]]
# [cost, value]
exp_node_weights = [[1, 8], [1, 1], [9, 11], [3, 2]]

budget = no_nodes*3 # example budget

###############################################################################


#### NAIVE SOLUTION ###########################################################

#### BRUTE FORCE ALL COMBINATIONS #############################################
def combinations(graph, all_combs, all_costs, all_values, start, prev_cost, prev_value, prev_nodes):
    """Bottom up DP calculation of values and costs for all 2**n-1 node combinations

    Args:
        start (int): Description
        prev_cost (int): Description
        prev_value (int): Description
        prev_nodes (list of int): Description
    """
    for ii in range(start, graph.size):
        # combination
        nodes = prev_nodes + [ii]
        all_combs.append(nodes)
        # cost
        cost = prev_cost + graph.node_weights[ii][0]
        all_costs.append(cost)
        # value
        value = prev_value + graph.node_weights[ii][1] - graph.node_weights[ii][0]
        for node in prev_nodes: # complementarity
            for adjacent in graph.graph[node]:
                if adjacent[0] == ii:
                    value += adjacent[1]
        all_values.append(value)
        # recurse
        combinations(graph, all_combs, all_costs, all_values, ii+1, cost, value, nodes)

def get_best_combination(graph, budget):
    all_combs = []
    all_costs = []
    all_values = []
    combinations(graph, all_combs, all_costs, all_values, 0, 0, 0, [])

    #print(all_combs)
    #print(all_costs)
    #print(all_values)

    # find first winner that fits budget
    winners = sorted(((val, ind) for ind, val in enumerate(all_values)), reverse=True)
    print('\ncombinations')
    print('{} winners'.format(len(winners)))
    #print(all_combs[winners[0][1]])
    #print(winners[0][0])

    for winner in winners:
        cost = all_costs[winner[1]]
        if cost <= budget:
            print(all_combs[winner[1]])
            print(winner[0])
            break
###############################################################################

graph = RandomGraph('Gnp', 4, edge_spec=.5, seed=42)
budget = 100
get_best_combination(graph, budget)



#### GREEDY NODE QUALITY ######################################################
# sort nodes by their quality, which is their own value plus the sum of the values of all connecting edges
'''

# sort nodes by their quality, which is their own value plus the sum of the values of all connecting edges
node_qualities = []
for node in range(no_nodes):
    node_value = exp_node_weights[node][1] - exp_node_weights[node][0]
    edge_values = 0
    for adjacent in exp_graph[node]:
        edge_values += adjacent[1]
    node_quality = node_value + edge_values
    node_qualities.append(node_quality)

print('\nnode quality')
#print(node_qualities)
sorted_qualities = sorted(((val, ind) for ind, val in enumerate(node_qualities)), reverse=True)
#print(sorted_qualities)

# greedily add nodes in order of their quality
cost = 0
value = 0
node_set = set()
for node in sorted_qualities:
    if node[0] < 0: # only nodes of negative quality, i.e., cost, left
        break

    cost += exp_node_weights[node[1]][0]
    if cost > budget:
        break

    value += exp_node_weights[node[1]][1] - exp_node_weights[node[1]][0]
    node_set.add(node[1])
print(node_set)

# complementarity values
comp = 0
for node in node_set:
    for adjacent in exp_graph[node]:
        if adjacent[0] in node_set:
            comp += adjacent[1]
comp /= 2 # counted each edge twice
value += comp
print(value)
'''
###############################################################################



#### GREEDY MAXIMUM SPANNING TREE #############################################
def prim_MST():
    """Runs Prim's algorithm to find a maximum spanning tree (MST).

    Four modifications to standard minimum spanning tree:
        1) Edge weights are negated to go from minimum to maximum ST
        2) Node value is added to edge weight for consideration of candidate edges
        3) The value of the MST is not its length but the sum of all node values plus the sum of all edge values of edges between those nodes, including edges that are additional to the spanning tree
        4) The algorithm terminates if there are only nodes left that would have a negative impact on the current solution, i.e., infer a cost. Such termination might be preliminary, since addition of future (bad) nodes might be worthwhile due to complementarity effects.

    Args:
        s (int): Index of starting node

    Returns:
        tuple (set, int): ({node IDs in MST}, value of MST)
    """
    node_list = list(node_set)
    s = random.choice(node_list)

    val_s = -exp_node_weights[s][1] + exp_node_weights[s][0] # value of starting node
    value = 0 # MST value
    cost = 0 # MST cost
    add_cost = 0
    compensation = 0 # compensation for edges that are counted twice

    prev = [0]*no_nodes
    dist = [math.inf]*no_nodes
    S = set()
    H = Heap(no_nodes)
    H.insert(s, val_s, 0)
    dist[s] = val_s

    for v in range(no_nodes):
        H.insert(v, math.inf, 0)

    while H.size > 0:
        v = H.delete_min()
        if v[1] > 0: # min in Heap is of positive value, i.e., a cost, abort
            break

        # abort if out of budget
        cost += exp_node_weights[v[0]][0]
        if cost > budget:
            #print(value, compensation)
            MST_value = -(value-compensation)
            return (S, MST_value)

        # complementarity
        for node in S:
            for adjacent in exp_graph[node]:
                if adjacent[0] == v[0]:
                    value -= adjacent[1]

        S.add(v[0])
        value += v[1]
        compensation += v[2] # necessary since edge weight was added to node quality already

        for w in exp_graph[v[0]]:
            if not w[0] in S:
                if dist[w[0]] > w[1]:
                    d = -w[1] - exp_node_weights[w[0]][1] + exp_node_weights[w[0]][0] # negate for maximum spanning tree
                    if d > 0: # bad/costly node
                        continue
                    dist[w[0]] = d
                    comp = -w[1]
                    prev[w[0]] = v[0]
                    H.decrease_key(w[0], dist[w[0]], comp)

    MST_value = -(value-compensation)
    del H
    return (S, MST_value, cost, add_cost)
'''
# start with most promising node
best_val = -math.inf
best_ind = -math.inf
for ind, weight in enumerate(exp_node_weights):
    current_val = weight[1] - weight[0]
    if current_val > best_val:
        best_val = current_val
        best_ind = ind
'''
'''
print('\nspanning tree')
remaining_budget = budget
value = 0
nodes = set()
node_set = set(range(no_nodes))
add_cost = 0
while node_set and remaining_budget > 6:
    MST_nodes, MST_value, MST_cost, add_cost = prim_MST()
    value += MST_value
    nodes.update(MST_nodes)
    node_set -= MST_nodes
    remaining_budget -= MST_cost
    #print(value, remaining_budget)
    #print(nodes, node_set)
    #print('\n')
#print(nodes)
#print()
print(nodes)
print(value)
'''
###############################################################################


combos = 0
nodeqs = 0
strees = 0
for ii in range(10):
    #combos
    all_combs = []
    all_costs = []
    all_values = []
    combinations(graph, all_combs, all_costs, all_values, 0, 0, 0, [])
    # find first winner that fits budget
    winners = sorted(((val, ind) for ind, val in enumerate(all_values)), reverse=True)
    for winner in winners:
        cost = all_costs[winner[1]]
        if cost <= budget:
            combos += winner[0]
            break
    # nodeqs
    node_qualities = []
    for node in range(no_nodes):
        node_value = exp_node_weights[node][1] - exp_node_weights[node][0]
        edge_values = 0
        for adjacent in exp_graph[node]:
            edge_values += adjacent[1]
        node_quality = node_value + edge_values
        node_qualities.append(node_quality)
    sorted_qualities = sorted(((val, ind) for ind, val in enumerate(node_qualities)), reverse=True)
    # greedily add nodes in order of their quality
    cost = 0
    value = 0
    node_set = set()
    for node in sorted_qualities:
        if node[0] < 0: # only nodes of negative quality, i.e., cost, left
            break
        cost += exp_node_weights[node[1]][0]
        if cost > budget:
            break
        value += exp_node_weights[node[1]][1] - exp_node_weights[node[1]][0]
        node_set.add(node[1])
    # complementarity values
    comp = 0
    for node in node_set:
        for adjacent in exp_graph[node]:
            if adjacent[0] in node_set:
                comp += adjacent[1]
    comp /= 2 # counted each edge twice
    value += comp
    nodeqs += value
    # strees
    remaining_budget = budget
    value = 0
    nodes = set()
    node_set = set(range(no_nodes))
    add_cost = 0
    while node_set and remaining_budget > 6:
        MST_nodes, MST_value, MST_cost, add_cost = prim_MST()
        value += MST_value
        nodes.update(MST_nodes)
        node_set -= MST_nodes
        remaining_budget -= MST_cost

    strees += value
combos /= 10
nodeqs /= 10
strees /= 10
print(combos, nodeqs, strees)
