import pandas as pd
import networkx as nx


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
    graph = nx.Graph()

    for coauthor in coauthor_list:
        m = len(coauthor)
        for i in range(m):
            for j in range(i + 1, m):
                graph.add_edge(coauthor[i], coauthor[j])

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


if __name__ == '__main__':
    load_dataset()
