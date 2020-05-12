"""This library provides the data structures for Gnp and Gnm random graphs as well as grid graphs.
"""
import math
import random

class RandomGraph():
    """
    Class for Gnp and Gnm random graphs as well as grid graphs

    Attributes:
        edges (int): Counts the number of edges in the graph
        graph (list of lists): Represents the graph in adjecency list format
    """

    def __init__(self, graph_type, graph_size, edge_spec=0):
        """Generates graph

        Args:
            graph_type (string): Gnp or Gnm graph
            graph_size (int): Number of nodes n
            edge_spec (float): Edge probability p or number of edges m
        """
        self.graph_type = graph_type
        self.edge_spec = edge_spec
        self.size = graph_size
        self.graph = [[] for n in range(graph_size)] # adjacency list
        self.node_weights = [[random.randint(0, 10), random.randint(0, 10)] for n in range(graph_size)] # cost, benefit
        self.edges = 0
        if graph_type == 'Gnp':
            self.generate_Gnp(graph_size, edge_spec)
        elif graph_type == 'Gnm':
            self.generate_Gnm(graph_size, edge_spec)
        elif graph_type == 'grid':
            self.generate_grid(graph_size)
        else:
            print('Invalid graph type. Choose from Gnp and Gnm.')

    def __str__(self):
        out = 'graph type = {}, n = {}'.format(self.graph_type, len(self.graph))
        if self.graph_type == 'Gnp':
            out += ', p = {}'.format(self.edge_spec)
        elif self.graph_type == 'Gnm':
            out += ', m = {}'.format(self.edge_spec)
        out += '\n'
        out += 'graph: ' + str(self.graph) + '\n'
        out += 'node_weights: ' + str(self.node_weights) + '\n'
        out += 'edges: {}'.format(self.edges)
        return out

    def get_set_cost(self, nodes):
        """ for a set of nodes, compute the total cost
        (sum of cost of nodes) """

        cost = 0
        for v in nodes:
            cost += graph[v][0]

        return cost

    def get_set_benefit(self):
        """ for a set of nodes, compute the total benefit
        (sum of benefits of node, plus complementarity of edges) """

        benefit = 0

        # add cost/benefit of nodes
        for v in nodes:
            benefit += graph[v][1]

        # add benefit of edges
        edge_benefit = 0
        for v in nodes:
            for adj, weight in graph.graph[v]:
                if adj in nodes:
                    edge_benefit += weight

        benefit += edge_benefit / 2   # edges are double counted

        return benefit

    def generate_Gnp(self, n, p):
        """Gnp graph

        Args:
            n (int): Number of nodes
            p (float): Edge probability
        """
        for node in range(n):
            for neighbor in range(node+1, n):
                if random.random() < p: # < bcs 0 inclusive but 1 not
                    weight = random.randint(0, 11)
                    self.edges += 1
                    self.graph[node].append([neighbor, weight])
                    self.graph[neighbor].append([node, weight]) # undirected

    def generate_Gnm(self, n, m):
        """Gnm graph

        Args:
            n (int): Number of nodes
            m (int): Number of edges
        """
        def extract(lst):
            return [item[0] for item in lst]

        while self.edges < m: # add edges until graph complete
            while True: # sample until new edge
                nodes = random.sample(range(n), 2)
                if not nodes[1] in extract(self.graph[nodes[0]]):
                    self.edges += 1
                    weight = random.randint(0, 10)
                    self.graph[nodes[0]].append([nodes[1], weight])
                    self.graph[nodes[1]].append([nodes[0], weight]) # undirected
                    break

    def generate_grid(self, n):
        """Regular grid graph

        Args:
            n (int): Number of nodes
        """
        # number of nodes has to be the square of an integer
        side_length = math.floor(math.sqrt(n))
        no_nodes = side_length**2
        # overwrite graph and agents if size has changed
        if no_nodes != n:
            self.graph = [[] for n in range(no_nodes)] # adjacency list

        # bottom
        self.graph[0].append(1)
        self.graph[0].append(side_length)
        self.graph[side_length-1].append(side_length-2)
        self.graph[side_length-1].append(2*side_length-1)
        for n in range(1, side_length-1):
            self.graph[n].append(n-1) # left
            self.graph[n].append(n+1) # right
            self.graph[n].append(n+side_length) # above

        # top
        t = (side_length-1)*side_length
        self.graph[t].append(t+1)
        self.graph[t].append(t-side_length)
        self.graph[no_nodes-1].append(no_nodes-2)
        self.graph[no_nodes-1].append(no_nodes-1-side_length)
        for n in range(t+1, t+side_length-1):
            self.graph[n].append(n-1) # left
            self.graph[n].append(n+1) # right
            self.graph[n].append(n-side_length) # below

        # left
        for n in range(side_length, t, side_length):
            self.graph[n].append(n-side_length) # below
            self.graph[n].append(n+side_length) # above
            self.graph[n].append(n+1) # right

        # right
        for n in range(2*side_length-1, no_nodes-1, side_length):
            self.graph[n].append(n-side_length) # below
            self.graph[n].append(n+side_length) # above
            self.graph[n].append(n-1) # left

        # inside
        for row in range(1, side_length-1):
            for col in range(1, side_length-1):
                node = row*side_length + col
                self.graph[node].append(node-side_length) # below
                self.graph[node].append(node+side_length) # above
                self.graph[node].append(node-1) # left
                self.graph[node].append(node+1) # right
