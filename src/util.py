import random
import pandas as pd
import networkx as nx
from sklearn import preprocessing


def load_dataset():
    """
    Load raw dataset
    """
    print("Load Dataset ...")

    paper_author_data = []
    with open("../data/paper_author.txt", "r") as f:
        n, m = [int(k) for k in f.readline().split()]
        lines = f.readlines()
        assert len(lines) == m
        for line in lines:
            author_list = [int(k) for k in line.split()]
            for author in author_list:
                assert author <= n
            paper_author_data.append(author_list)

    public_true_data = []
    public_false_data = []

    with open("../data/query_public.txt", "r") as f:
        with open("../data/answer_public.txt", "r") as g:
            q = int(f.readline().strip())
            f_lines = f.readlines()
            g_lines = g.readlines()
            assert q == len(f_lines) and q == len(g_lines)

            for f_line, g_line in zip(f_lines, g_lines):
                author_list = [int(k) for k in f_line.split()]
                if g_line.strip() == "True":
                    public_true_data.append(author_list)
                else:
                    public_false_data.append(author_list)

    private_data = []
    with open("../data/query_private.txt", "r") as f:
        q = int(f.readline().strip())
        lines = f.readlines()
        assert q == len(lines)

        for line in lines:
            author_list = [int(k) for k in line.split()]
            private_data.append(author_list)

    true_data = paper_author_data + public_true_data

    print("Load Dataset Done!")

    print("# of total authors:", n)
    print("# of paper_author:", len(paper_author_data))
    print("# of public_true:", len(public_true_data))
    print("# of true:", len(true_data))
    print("# of public_false:", len(public_false_data))
    print("# of private:", len(private_data))

    return n, paper_author_data, public_true_data, public_false_data, private_data


def make_graph(coauthor_list):
    """
    Return networkx graph based on coauthor_list
    Index of author starts from 1
    """
    print("Make Graph ...")
    graph = nx.Graph()

    for coauthor in coauthor_list:
        m = len(coauthor)
        for i in range(m):
            for j in range(i + 1, m):
                graph.add_edge(coauthor[i], coauthor[j])

    print("Make Graph Done!")
    return graph


def make_tabular(n_total, coauthor_list, position_encode=False):
    """
    Index of author starts from 1
    """

    m_total = len(coauthor_list)

    author_dict = {}
    for i in range(n_total):
        author_name = "a{}".format(i + 1)
        author_dict[author_name] = [0] * m_total

    for i, coauthors in enumerate(coauthor_list):
        for j, author in enumerate(coauthors):
            author_name = "a{}".format(author)
            if position_encode:
                author_dict[author_name][i] = j + 1
            else:
                author_dict[author_name][i] = 1

    return pd.DataFrame(data=author_dict)


def get_two_random(a, b, ordering=True):
    """
    Return two distinct random integers i, j in [a, b] s.t. i < j.
    """
    i, j = 0, 0
    while i == j:
        i = random.randint(a, b)
        j = random.randint(a, b)

    if ordering:
        return min(i, j), max(i, j)
    else:
        return i, j


def partition(list_in, n):
    """
    Partition input list into n chunks in nearly equal size
    """
    list_copy = list_in.copy()
    random.shuffle(list_copy)
    return [list_copy[i::n] for i in range(n)]


def compute_accuracy(true_labels, predictions):
    tp, tn, fp, fn = 0, 0, 0, 0
    assert len(true_labels) == len(predictions)
    for i in range(len(true_labels)):
        if true_labels[i] == 1 and predictions[i] == 1:
            tp += 1
        elif true_labels[i] == 1 and predictions[i] == 0:
            fn += 1
        elif true_labels[i] == 0 and predictions[i] == 1:
            fp += 1
        elif true_labels[i] == 0 and predictions[i] == 0:
            tn += 1
        else:
            raise Exception("wrong labels: {}, {}".format(
                true_labels[i], predictions[i]))

    print(tp, tn, fp, fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    print("acc: {}%".format(round(acc * 100, 2)))


def means(_list_1d):
    list_1d = [x for x in _list_1d if x != 0]
    if len(list_1d) == 0:
        return [0, 0, 0]

    am, gm, hm = 0, 1, 0
    for l in list_1d:
        assert l != 0
        am += l
        gm *= pow(l, 1 / len(list_1d))
        hm += 1 / l

    am /= len(list_1d)
    if hm > 0:
        hm = len(list_1d) / hm

    return [am, gm, hm]


def standardize(feature_matrix):
    df = pd.DataFrame(feature_matrix)
    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(df)
    return standard_scaler.transform(df)


if __name__ == '__main__':
    load_dataset()
