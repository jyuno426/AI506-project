from node_embedding import build_node_embedding
from util import load_dataset, make_graph

from moea import remove_subset
import numpy as np
import random
import networkx as nx
import json

if __name__ == '__main__':
    # n, paper_author_data, public_true_data, public_false_data, private_data = load_dataset()

    
    graph = nx.fast_gnp_random_graph(n=101, p=0.5)
    graph.remove_node(0)

    # node name should be sequential from 1

    # to compute l of d_value in evalutation
    adj_dict = nx.to_dict_of_dicts(graph)
    with open("../output/adj_dict" + ".json", "w") as f:
        json.dump(adj_dict, f)
    #np.save("../output/adj_dict",adj_mat)
    
    # compute co-author set
    co_author = list()
    for k in adj_dict.keys():
        idxs = list(adj_dict[k].keys())
        co_author.append(idxs)
    '''
    for i in range(len( adj_mat)):
        idxs = np.nonzero(adj_mat[i,:])[0] +1
        co_set = np.append(idxs, i+1).tolist()
        co_author.append(co_set)
    '''

    co_author = remove_subset(co_author)
    np.save("../output/co_author",np.array(co_author))

    ## make validation data
    
    # validation = random.sample(co_author, 10)
    # print(validation)

    ## make test data
    test = random.sample(co_author, 10)
    f = open("../output/test.txt", 'w')

    for line in test:
        new = random.sample(line,random.randint(1,int(len(test)/2)))
        new = list(map(str, new))
        newstr = " ".join(new)+'\n'
        f.write(newstr)

    f.close()
    
    build_node_embedding(graph, dimensions=10, p=1, q=0.5,
                         walk_length=10, num_walks=50, workers=1)
    build_node_embedding(graph, dimensions=10, p=1, q=2,
                         walk_length=10, num_walks=50, workers=1)
    