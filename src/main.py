from node_embedding import build_node_embedding
from util import load_dataset, make_graph


if __name__ == '__main__':
    n, paper_author_data, public_true_data, public_false_data, private_data = load_dataset()
    graph = make_graph(paper_author_data + public_true_data)

    # import networkx as nx
    # graph = nx.fast_gnp_random_graph(n=101, p=0.5)
    # graph.remove_node(0)

    build_node_embedding(graph, dimensions=10, p=1, q=0.5,
                         walk_length=10, num_walks=50, workers=10)
    build_node_embedding(graph, dimensions=10, p=1, q=2,
                         walk_length=10, num_walks=50, workers=10)
