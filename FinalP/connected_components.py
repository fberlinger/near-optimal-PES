from collections import deque
from lib_randomgraph import RandomGraph
import numpy as np

from max_spanning_tree import spanning_tree

def main():
    graph = RandomGraph('Gnp', 4, edge_spec=.5, seed=42)

    print(graph)
    connected_components(graph)

    print('num components: {}'.format(len(components)))
    for i, component in enumerate(components):
        print('  component {}, size {}, {}'.format(i, len(component), component))
    print('benefit - cost = net benefit')
    for i, component in enumerate(components):
        benefit = graph.get_set_benefit(component)
        cost = graph.get_set_cost(component)
        print('  component {}, {} - {} = {}'.format(i, benefit, cost, benefit-cost))


def connected_components(graph):
    """ discover connected components with BFS """

    if graph.size == 0:
        return []

    components = [[]]

    not_visited = set(range(graph.size))

    # retrieve random element from set and add to queue
    v = next(iter(not_visited))
    q = deque()
    q.append(v)

    while not_visited:
        if q:
            # if the queue has items, we continue to grow our connected component
            v = q.popleft()
            components[-1].append(v)
        else:
            # if the queue is empty, we start a new connected component
            v = next(iter(not_visited))
            components.append([v])

        # add all neighbors of this node
        for adj, weight in graph.graph[v]:
            if adj in not_visited:
                q.append(adj)
        not_visited.remove(v)

    return components

def max_component(graph, budget):
    components = connected_components(graph)
    values = []
    for i, component in enumerate(components):
        benefit = graph.get_set_benefit(component)
        cost = graph.get_set_cost(component)
        print('  component {}, {} - {} = {}'.format(i, benefit, cost, benefit-cost))
        values.append(benefit - cost)

    max_vals = reversed(np.argsort(values))

    # selected parcels
    selected = set()
    remaining_budget = budget

    # use up all our budget
    for max_idx in max_vals:  # iterate through most valuable parcels
        set_cost = graph.get_set_cost(components[max_idx])
        if set_cost <= remaining_budget:
            selected.update(components[max_idx])
            remaining_budget -= set_cost
        else:
            # create map for nodes
            subgraph_map = {}  # {new_node: old_node}
            for i in range(len(components[max_idx])):
                subgraph_map[i] = components[max_idx][i]

            # compute a MST through this to use up as much as possible
            subgraph = RandomGraph('manual', len(components[max_idx]))

            # build adjacency list
            for i in range(len(components[max_idx])):
                subgraph.graph[i] += graph.graph[subgraph_map[i]]

            subgraph.node_weights = [[graph.node_weights[subgraph_map[i]][0], graph.node_weights[subgraph_map[i]][1]] for i in range(len(components[max_idx]))]

            # subgraph.edges = 0

            sub_nodes, sub_value = spanning_tree(subgraph, remaining_budget)

            # map selected nodes back to real numbers, add to our selected set, and update budget
            for selected_node in sub_nodes:
                selected.add(subgraph_map[selected_node])
                remaining_budget -= graph.node_weights[subgraph_map[selected_node]][0]

    return selected


if __name__ == '__main__':
    main()
