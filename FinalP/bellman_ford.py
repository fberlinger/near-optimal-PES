from lib_randomgraph import RandomGraph

def main():
    graph = RandomGraph('Gnp', 4, edge_spec=.5, seed=42)

    print(graph)
    dist, pred = bellman_ford(graph, 1, 10)
    print('distance    ', dist)
    print('predecessor ', pred)



def bellman_ford(graph, s, budget):
    """ compute shortest paths from a single node, in terms of cost
    negative weight edges ok

    v - source vertex """

    dist = {}   # distance from v
    pred = {}       # predecessor

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


if __name__ == '__main__':
    main()
