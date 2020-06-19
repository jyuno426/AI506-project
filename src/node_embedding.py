import json
from node2vec import Node2Vec


def build_node_embedding(graph, dimensions=30, p=1, q=1, walk_length=10, num_walks=50, workers=4, temp_folder=None):
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

    filepath = "../output/node_embedding_p{}_q{}".format(p, q)

    # Save embeddings for later use
    model.wv.save_word2vec_format(filepath)
    node_embedding_text_to_json(filepath)

    # Save model for later use
    # model.save("../model/node_embedding")


def node_embedding_text_to_json(filepath):
    res = {}
    with open(filepath, "r") as f:
        f.readline()
        for line in f.readlines():
            elems = line.split()
            res[int(elems[0])] = [float(x) for x in elems[1:]]

    with open(filepath + ".json", "w") as f:
        json.dump(res, f)


if __name__ == '__main__':
    node_embedding_text_to_json("../output/node_embedding_p1_q0.5")
    node_embedding_text_to_json("../output/node_embedding_p1_q2")
