import math
import random
import time

from lib_randomgraph import RandomGraph

from itertools import combinations

'''
graph_type = 'Gnm'
graph_size = 4 # n
no_edges = 4 # m
G = RandomGraph(graph_type, graph_size, no_edges)

print(G.graph)
print(G.node_weights)


graph_size = 4 # n

exp_graph = [[[1, 4], [2, 7]], [[2, 1], [0, 4], [3, 9]], [[1, 1], [0, 7]], [[1, 9]]]
exp_node_weights = [[1, 8], [1, 1], [9, 11], [3, 2]]

print(exp_graph)
print(exp_node_weights)

# brute force all solutions
all_combs = []
all_values = []
all_costs = []
all_nodes = range(graph_size)

no_combs = 0

for no_nodes in range(1, graph_size+1):
	current_combs = list(combinations(all_nodes, no_nodes))
	while current_combs:
		comb = current_combs.pop()
		all_combs.append(comb)
		if len(comb) == 1:
			cost = exp_node_weights[comb[0]][0]
			benefit = exp_node_weights[comb[0]][1]
			all_values.append(benefit-cost)
			all_costs.append(cost)
		else:
			cost = all_costs[-no_combs] + exp_node_weights[comb[0]][0]
			all_costs.append(cost)
			#benefit = all #xx add edge_weights

		no_combs += 1

print(all_combs)
print(all_values)
print(all_costs)
'''

no_nodes = 4 # n
#current_combs = all_combs.copy()
#for node in current_combs:
#	for new in range(node+1, no_nodes):
#		all_combs.append((node, new))
#print(all_combs)


def combinations(no_nodes):
	all_combs = [[i] for i in range(no_nodes)]
	
	for comb in all_combs:
		for node in range(comb[-1]+1, no_nodes):
			print(node)
			all_combs.append(comb + [node])
	
	return all_combs

all_combs = combinations(no_nodes)

print(all_combs)