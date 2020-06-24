import json
import random
import numpy as np
import networkx as nx

from moea import NMOEA
from node_embedding import build_node_embedding
from util import load_dataset, make_graph

random.seed(0)
np.random.seed(0)

if __name__ == '__main__':
    # Node name should be sequential from 1

    # Load dataset
    n, paper_author_data, public_true_data, public_false_data, test = load_dataset()
    true_data = paper_author_data + public_true_data
    random.shuffle(true_data)

    # Split train and valid sets
    valid_size = len(public_false_data)
    false_valid = public_false_data
    true_valid = true_data[:valid_size]
    true_train = true_data[valid_size:]

    # Make real graph
    graph = make_graph(true_train)

    # print(graph.number_of_edges())

    # Build node embedding
    s_dict = build_node_embedding(graph, dimensions=20, p=1, q=0.5,
                                  walk_length=80, num_walks=10, workers=10)
    d_dict = build_node_embedding(graph, dimensions=10, p=1, q=2,
                                  walk_length=80, num_walks=10, workers=10)

    # Run NMOEA
    nmoea = NMOEA(graph, s_dict, d_dict)
    nmoea.run()

    """
    # Make random graph
    print("Make Random Graph ...")
    graph = nx.fast_gnp_random_graph(n=101, p=0.5)
    graph.remove_node(0)
    print("Make Random Graph Done!")
    
    # to compute l of d_value in evalutation
    print("Compute adj_dict ...")
    adj_dict = nx.to_dict_of_dicts(graph)
    with open("../output/adj_dict" + ".json", "w") as f:
        json.dump(adj_dict, f)

    # compute co-author set
    print("Compute co_author set ...")
    co_author = list()
    for k in adj_dict.keys():
        idxs = list(adj_dict[k].keys())
        co_author.append(idxs)

    co_author = remove_subset(co_author)
    np.save("../output/co_author", np.array(co_author))
    """

    """
    # make validation data

    # validation = random.sample(co_author, 10)
    # print(validation)

    # make test data
    test = random.sample(co_author, 10)
    f = open("../output/test.txt", 'w')

    for line in test:
        new = random.sample(line, random.randint(1, int(len(test)/2)))
        new = list(map(str, new))
        newstr = " ".join(new)+'\n'
        f.write(newstr)

    f.close()
    """
