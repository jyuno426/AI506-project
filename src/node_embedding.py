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
    model = node2vec.fit(
        window=10, min_count=1, batch_words=4
    )  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Save embeddings for later use
    model.wv.save_word2vec_format(
        "./output/node_embedding_p{}_q{}".format(p, q))

    # Save model for later use
    model.save("./model/node_embedding")
