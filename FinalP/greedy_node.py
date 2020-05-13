from lib_randomgraph import RandomGraph

def main():
    graph = RandomGraph('Gnp', 4, edge_spec=.5, seed=42)

    print(graph)
    budget = 100
    nodes, value = greedy_node(graph, budget)
    print('selected nodes ', nodes)
    print('total value    ', value)

def greedy_node(graph, budget):
    # sort nodes by their quality, which is their own value plus the sum of the values of all connecting edges

    # sort nodes by their quality, which is their own value plus the sum of the values of all connecting edges
    node_qualities = []
    for node in range(graph.size):
        node_value = graph.node_weights[node][1] - graph.node_weights[node][0]
        edge_values = 0
        for adjacent in graph.graph[node]:
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

        cost += graph.node_weights[node[1]][0]
        if cost > budget:
            break

        value += graph.node_weights[node[1]][1] - graph.node_weights[node[1]][0]
        node_set.add(node[1])

    # complementarity values
    comp = 0
    for node in node_set:
        for adjacent in graph.graph[node]:
            if adjacent[0] in node_set:
                comp += adjacent[1]
    comp /= 2 # counted each edge twice
    value += comp
    
    return (node_set, value)


if __name__ == '__main__':
    main()
