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
            # print('from queue', v)
        else:
            # if the queue is empty, we start a new connected component
            v = next(iter(not_visited))
            components.append([v])
            # print('new component', v)

        # add all neighbors of this node
        for adj, weight in graph.graph[v]:
            if adj in not_visited and adj not in q:
                q.append(adj)
        # print(not_visited, q)
        not_visited.discard(v)

    # all items should be in a component
    assert np.sum([len(component) for component in components]) == graph.size

    return components

def max_component(graph, budget):
    components = connected_components(graph)

    # order components in terms of lowest cost/benefit ratio
    ratios = []
    for i, component in enumerate(components):
        benefit = graph.get_set_benefit(component)
        cost = graph.get_set_cost(component)
        print('  component {}, size {}, {} - {} = {}'.format(i, len(component), benefit, cost, benefit-cost))
        # values.append(benefit - cost)
        ratios.append(cost / benefit)

    # selected parcels
    selected = set()
    remaining_budget = budget

    # use up all our budget
    for idx in np.argsort(ratios):  # iterate through most valuable parcels
        set_cost = graph.get_set_cost(components[idx])
        if set_cost <= remaining_budget:
            selected.update(components[idx])
            remaining_budget -= set_cost
        else:
            # create map for nodes
            v_to_subgraph = {}   # {old_node: new_node}
            subgraph_to_v = {}   # {new_node: old_node}
            for i in range(len(components[idx])):
                subgraph_to_v[i] = components[idx][i]
                v_to_subgraph[components[idx][i]] = i

            # compute a MST through this to use up as much as possible
            subgraph = RandomGraph('manual', len(components[idx]))

            # build adjacency list
            for i in range(len(components[idx])):
                subgraph.graph[i] = [[v_to_subgraph[adj], weight] for adj, weight in graph.graph[subgraph_to_v[i]]]

            subgraph.node_weights = [[v_to_subgraph[graph.node_weights[subgraph_to_v[i]][0]], graph.node_weights[subgraph_to_v[i]][1]] for i in range(len(components[idx]))]

            # subgraph.edges = 0

            sub_nodes, sub_value = spanning_tree(subgraph, remaining_budget)

            # map selected nodes back to real numbers, add to our selected set, and update budget
            for selected_node in sub_nodes:
                node_cost = graph.node_weights[subgraph_to_v[selected_node]][0]
                if node_cost <= remaining_budget: # TODO: this shouldn't be necessary; spanning tree should already account for this
                    selected.add(subgraph_to_v[selected_node])
                    remaining_budget -= node_cost

    return selected


if __name__ == '__main__':
    main()
