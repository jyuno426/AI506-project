import json
import random
import numpy as np
import networkx as nx

from moea import NMOEA
from node_embedding import build_node_embedding
from util import load_dataset, make_graph, compute_accuracy
import mglearn

random.seed(0)
np.random.seed(0)


if __name__ == '__main__':
    # Node name should be sequential from 1

    # Load dataset
    n, paper_author_data, public_true_data, public_false_data, test = load_dataset()
    true_data = paper_author_data + public_true_data
    random.shuffle(true_data)
    random.shuffle(public_false_data)

    # Split graph_making, train and valid sets
    train_size = int(len(public_false_data) * 0.7)
    valid_size = len(public_false_data) - train_size
    false_train = public_false_data[: train_size]
    false_valid = public_false_data[train_size:]
    true_train = true_data[: train_size]
    true_valid = true_data[train_size: train_size + valid_size]
    true_graph_making = true_data[valid_size:]

    # Make real graph
    graph = make_graph(true_graph_making)

    # Build node embedding
    s_dict = build_node_embedding(graph, dimensions=20, p=1, q=0.5,
                                  walk_length=80, num_walks=10, workers=10)
    d_dict = build_node_embedding(graph, dimensions=10, p=1, q=2,
                                  walk_length=80, num_walks=10, workers=10)

    # Run NMOEA
    nmoea = NMOEA(n, graph, s_dict, d_dict)
    # nmoea.run()

    with open("../iter500.json") as f:
        population = json.load(f)

    train_labels = [1] * len(true_train) + [0] * len(false_train)
    valid_labels = [1] * len(true_valid) + [0] * len(false_valid)
    train_set = true_train + false_train
    valid_set = true_valid + false_valid

    print("Train:")
    train_pred_labels = nmoea.fit(population, train_set, train_labels)
    compute_accuracy(train_labels, train_pred_labels)

    #plot_decision_function(train_set, train_labels, valid_set, valid_labels, nmoea.predictor)

    print("Valid:")
    valid_pred_labels = nmoea.eval(population, valid_set)
    compute_accuracy(valid_labels, valid_pred_labels)

    print("Test:")
    test_pred_labels = nmoea.eval(population, test)
    with open("../output/answer_private.txt", "w") as f:
        for label in test_pred_labels:
            if label == 1:
                f.write("False\n")
            elif label == 0:
                f.write("True\n")
            else:
                raise Exception("wrong label!!{}".format(label))
