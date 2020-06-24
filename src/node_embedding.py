import os
import json
from node2vec import Node2Vec


def build_node_embedding(graph, p=1, q=1, dimensions=30, walk_length=10, num_walks=50, workers=4, temp_folder=None):
    """
    Build node embedding using Node2Vec Library which is a wrapper of gensim word2vec.
    """

    if not os.path.exists("../output/node_embedding"):
        os.makedirs("../output/node_embedding")

    filepath = "../output/node_embedding/p{}_q{}_dim{}_wl{}_nw{}".format(
        p, q, dimensions, walk_length, num_walks)

    print("Build Node embedding at {} ...".format(filepath))

    if not os.path.exists(filepath):
        # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
        node2vec = Node2Vec(graph,
                            p=p,
                            q=q,
                            dimensions=dimensions,
                            walk_length=walk_length,
                            num_walks=num_walks,
                            workers=workers,
                            temp_folder=temp_folder)  # Use temp_folder for big graphs

        # Embed nodes
        # Any keywords acceptable by gensim.Word2Vec can be passed,
        # `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # Save model for later use (but too slow to save it)
        # model.save("../model/node_embedding")

        # Save embeddings
        model.wv.save_word2vec_format(filepath)

    res = node_embedding_text_to_dict(filepath)

    print("Build Node embedding Done!")
    return res


def node_embedding_text_to_dict(filepath):
    res = {}
    with open(filepath, "r") as f:
        f.readline()
        for line in f.readlines():
            elems = line.split()
            res[int(elems[0])] = [float(x) for x in elems[1:]]

    return res
