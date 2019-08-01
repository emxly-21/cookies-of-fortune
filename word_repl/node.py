import numpy as np
from random import randint
from collections import Counter
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import operator

class Node:
    """ Describes a node in a graph, and the edges connected
        to that node."""

    def __init__(self, ID, neighbors, descriptor, truth=None, file_path=None):
        """ Parameters
            ----------
            ID : int
                A unique identifier for this node. Should be a
                value in [0, N-1], if there are N nodes in total.
            neighbors : Sequence[int]
                The node-IDs of the neighbors of this node.
            descriptor : numpy.ndarray
                The (128,) descriptor vector for this node's picture
            truth : Optional[str]
                If you have truth data, for checking your clustering algorithm,
                you can include the label to check your clusters at the end.
                If this node corresponds to a picture of Ryan, this truth
                value can just be "Ryan"
            file_path : Optional[str]
                The file path of the image corresponding to this node, so
                that you can sort the photos after you run your clustering
                algorithm
            """
        self.id = ID  # a unique identified for this node - this should never change

        # The node's label is initialized with the node's ID value at first,
        # this label is then updated during the whispers algorithm
        self.label = ID

        # (n1_ID, n2_ID, ...)
        # The IDs of this nodes neighbors. Empty if no neighbors
        self.neighbors = tuple(neighbors)
        self.descriptor = descriptor

        self.truth = truth
        self.file_path = file_path


'''def plot_graph(graph, adj):
    """ Use the package networkx to produce a diagrammatic plot of the graph, with
        the nodes in the graph colored according to their current labels.
        Note that only 20 unique colors are available for the current color map,
        so common colors across nodes may be coincidental.
        Parameters
        ----------
        graph : Tuple[Node, ...]
            The graph to plot
        adj : numpy.ndarray, shape=(N, N)
            The adjacency-matrix for the graph. Nonzero entries indicate
            the presence of edges.
        Returns
        -------
        Tuple[matplotlib.fig.Fig, matplotlib.axis.Axes]
            The figure and axes for the plot."""
    import networkx as nx
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    g = nx.Graph()
    for n, node in enumerate(graph):
        g.add_node(n)

    # construct a network-x graph from the adjacency matrix: a non-zero entry at adj[i, j]
    # indicates that an egde is present between Node-i and Node-j. Because the edges are 
    # undirected, the adjacency matrix must be symmetric, thus we only look ate the triangular
    # upper-half of the entries to avoid adding redundant nodes/edges
    g.add_edges_from(zip(*np.where(np.triu(adj) > 0)))

    # we want to visualize our graph of nodes and edges; to give the graph a spatial representation,
    # we treat each node as a point in 2D space, and edges like compressed springs. We simulate
    # all of these springs decompressing (relaxing) to naturally space out the nodes of the graph
    # this will hopefully give us a sensible (x, y) for each node, so that our graph is given
    # a reasonable visual depiction 
    pos = nx.spring_layout(g)

    # make a mapping that maps: node-lab -> color, for each unique label in the graph
    color = list(iter(cm.tab20b(np.linspace(0, 1, len(set(i.label for i in graph))))))
    color_map = dict(zip(sorted(set(i.label for i in graph)), color))
    colors = [color_map[i.label] for i in graph]  # the color for each node in the graph, according to the node's label

    # render the visualization of the graph, with the nodes colored based on their labels!
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(g, pos=pos, ax=ax, nodelist=range(len(graph)), node_color=colors)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=g.edges())
    return fig, ax'''

def whispers(arr, numTimes):
    '''
    Given array of Nodes, perform the
    Whispers algorithm to group nodes.

    Parameters:
    -----------
    arr: list of Nodes
    numTimes: int

    Returns:
    --------
    list of Nodes, but colored
    post-Whispers algorithm (the labels)
    '''

    N = len(arr)

    for i in range(numTimes):
        cur_node = randint(0, N-1)
        defdct = defaultdict(int)
        for adj_node in arr[cur_node].neighbors:
            sim = cosine_similarity(arr[cur_node].descriptor.reshape(1,-1), arr[adj_node].descriptor.reshape(1,-1))[0]
            defdct[arr[adj_node].label] += sim
        if(len(defdct) != 0):
            dct = dict(defdct)
            # print(dct)
            most_com = max(dct, key=dct.get)
            arr[cur_node].label = most_com
        
        if(i % 200000 == 0):
            print(i)

    return arr