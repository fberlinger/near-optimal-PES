import math
import random

from lib_randomgraph import RandomGraph
from lib_heap import Heap

def main():
    graph = RandomGraph('Gnp', 20, edge_spec=.5)

    print(graph)
    budget = 40
    nodes, value = spanning_tree(graph, budget)
    print('selected nodes ', nodes)
    print('total value    ', value)

def prim_MST(graph, budget, node_set):
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

    val_s = -graph.node_weights[s][1] + graph.node_weights[s][0] # value of starting node
    value = 0 # MST value
    cost = 0 # MST cost
    compensation = 0 # compensation for edges that are counted twice

    prev = [0]*graph.size
    dist = [math.inf]*graph.size
    S = set()
    H = Heap(graph.size)
    H.insert(s, val_s, 0)
    dist[s] = val_s

    for v in range(graph.size):
        H.insert(v, math.inf, 0)

    while H.size > 0:
        v = H.delete_min()
        if v[1] > 0: # min in Heap is of positive value, i.e., a cost, abort
            break

        # abort if out of budget
        cost += graph.node_weights[v[0]][0]
        if cost > budget:
            MST_value = -(value-compensation)
            return (S, MST_value, cost-graph.node_weights[v[0]][0])

        # complementarity
        for node in S:
            for adjacent in graph.graph[node]:
                if adjacent[0] == v[0]:
                    value -= adjacent[1]

        S.add(v[0])
        value += v[1]
        compensation += v[2] # necessary since edge weight was added to node quality already

        for adj, weight in graph.graph[v[0]]:
            if not adj in S:
                if dist[adj] > weight:
                    d = -weight - graph.node_weights[adj][1] + graph.node_weights[adj][0] # negate for maximum spanning tree
                    if d > 0: # bad/costly node
                        continue
                    dist[adj] = d
                    comp = -weight
                    prev[adj] = v[0]
                    H.decrease_key(adj, dist[weight], comp)

    MST_value = -(value-compensation)
    del H
    print('return', S, MST_value, cost)
    return (S, MST_value, cost)

def spanning_tree(graph, budget):
    remaining_budget = budget
    value = 0
    nodes = set()
    node_set = set(range(graph.size))

    while node_set and remaining_budget > 6:
        MST_nodes, MST_value, MST_cost = prim_MST(graph, remaining_budget, node_set)
        value += MST_value
        nodes.update(MST_nodes)
        node_set -= MST_nodes
        remaining_budget -= MST_cost

    return (nodes, value)


if __name__ == '__main__':
    main()


'''
# start with most promising node
best_val = -math.inf
best_ind = -math.inf
for ind, weight in enumerate(graph.node_weights):
    current_val = weight[1] - weight[0]
    if current_val > best_val:
        best_val = current_val
        best_ind = ind
'''
