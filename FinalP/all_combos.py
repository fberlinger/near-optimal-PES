from lib_randomgraph import RandomGraph

def main():
    graph = RandomGraph('Gnp', 4, edge_spec=.5, seed=42)

    print(graph)
    budget = 100
    nodes, value = get_best_combination(graph, budget)
    print('selected nodes ', nodes)
    print('total value    ', value)

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

    # find first winner that fits budget
    winners = sorted(((val, ind) for ind, val in enumerate(all_values)), reverse=True)

    for winner in winners:
        cost = all_costs[winner[1]]
        if cost <= budget:
            return (all_combs[winner[1]], winner[0])

if __name__ == '__main__':
    main()
