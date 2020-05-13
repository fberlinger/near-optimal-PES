from lib_randomgraph import RandomGraph
import numpy as np

def main():
    graph = RandomGraph('Gnp', 4, edge_spec=.5, seed=42)

    print(graph)
    dist, pred = bellman_ford(graph, 1, 10)
    print('distance    ', dist)
    print('predecessor ', pred)


def bellman_ford(graph, s, budget):
    """ compute shortest paths from a single node, in terms of cost
    negative weight edges ok

    s - source vertex """

    dist = {}   # distance from s
    pred = {}   # predecessor

    # initialize graph
    for v in range(graph.size):
        dist[v] = 2e10

    dist[s] = graph.node_weights[s][0]  # distance from source to itself is its cost
    pred[s] = None

    # relax edges repeatedly
    for _ in range(graph.size - 1):  # V-1 iterations
        for v1 in range(graph.size):
            for v2, edge_weight in graph.graph[v1]:
                node_weight = graph.node_weights[v2][0]
                if dist[v1] + node_weight < dist[v2]:
                    dist[v2] = dist[v1] + node_weight
                    pred[v2] = v1

    # check for negative-weight cycles
    # if any negative-weight cycles, not valid

    return dist, pred


def select_bellman_ford(graph, budget):
    # start with highest value node
    node_values = graph.get_node_values()
    node_costs = graph.get_node_costs()
    max_val = np.argsort(node_values)[-1]
    dist, pred = bellman_ford(graph, max_val, budget)

    selected = set()
    remaining_budget = budget
    smallest_dist = np.argsort([dist[v] for v in np.arange(graph.size)])
    # print('dist', dist)
    # print('smallest', smallest_dist)
    
    # iterate from smallest-cost node through remaining nodes until budget is full
    for min_idx in smallest_dist:
        if remaining_budget >= node_costs[min_idx]:
            selected.add(min_idx)
            remaining_budget -= node_costs[min_idx]

    return selected

if __name__ == '__main__':
    main()
