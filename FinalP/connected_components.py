from collections import deque
from lib_randomgraph import RandomGraph

def main():
    graph = RandomGraph('Gnp', 4, edge_spec=.5, seed=42)

    print(graph)
    connected_component(graph)


def connected_component(graph):
    """ discover connected components with BFS """

    if graph.size == 0:
        return []

    components = [[]]

    not_visited = set(range(graph.size))

    v = next(iter(not_visited))  # retrieve random element from set
    q = deque()
    q.append(v)

    while not_visited:
        # print(not_visited)
        if q:
            v = q.popleft()
            components[-1].append(v)
        else:
            v = next(iter(not_visited))
            components.append([v])
        # print('  ', v)

        for adj, weight in graph.graph[v]:
            if adj in not_visited:
                q.append(adj)
        not_visited.remove(v)

    print('num components: {}'.format(len(components)))
    for i, component in enumerate(components):
        print('  component {}, size {}, {}'.format(i, len(component), component))
    print('benefit - cost = net benefit')
    for i, component in enumerate(components):
        benefit = graph.get_set_benefit(component)
        cost = graph.get_set_cost(component)
        print('  component {}, {} - {} = {}'.format(i, benefit, cost, benefit-cost))

    return components


if __name__ == '__main__':
    main()
