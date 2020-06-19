# from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from util import load_dataset


def set_to_str(_set):
    _list = list(_set)
    _list.sort()
    return "_".join([str(s) for s in _list])


def get_ones(bits):
    res = []
    idx = 0
    while bits > 0:
        if bits % 2 == 1:
            res.append(idx)
        bits //= 2
        idx += 1
    return res


def select_subset(_set, _n):
    assert isinstance(_set, set)
    # assert len(_set) >= _n
    # assert _n <= 2

    if len(_set) < _n:
        return []

    res = []

    _list = list(_set)
    m = len(_list)
    if _n == 1:
        for s in _list:
            res.append(set([s]))
    elif _n == 2:
        for i in range(m):
            for j in range(i + 1, m):
                res.append(set([_list[i], _list[j]]))
    elif _n == 3:
        for i in range(m):
            for j in range(i + 1, m):
                for k in range(j + 1, m):
                    res.append(set([_list[i], _list[j], _list[k]]))
    # for i in range((1 << m)):
    #     ones = get_ones(i)
    #     if len(ones) == _n:
    #         res.append(set([_list[idx] for idx in ones]))

    return res


class Expansion:
    def __init__(self, n, h_edges):
        self.n = n
        self.h_edges = h_edges
        self.subset_cnt = {}

        m = len(h_edges)
        for i, h_edge in enumerate(h_edges):
            print(i + 1, "/", m, "len:", len(h_edge))
            for subset_str in select_subset(h_edge, n):
                ss = set_to_str(subset_str)
                if ss in self.subset_cnt:
                    self.subset_cnt[ss] += 1
                else:
                    self.subset_cnt[ss] = 1

    def is_edge(self, u, v):
        """
        @params: {
            u: n order vertex
            v: n order vertex
        }
        """
        assert isinstance(u, set)
        assert isinstance(v, set)
        assert len(u) == self.n - 1
        assert len(v) == self.n - 1

        subset = u.union(v)
        return len(subset) == self.n and set_to_str(subset) in self.subset_cnt

    def get_weight(self, edge):
        """
        @params:{
            edge: n order edge
        }
        """
        w = 0
        subset = edge[0].union(edge[1])

        # assert len(subset) == 2
        s = set_to_str(subset)
        w = self.subset_cnt[s]

        # for h_edge in self.h_edges:
        #     if subset.issubset(h_edge):
        #         w += 1
        return w

    def get_inner_edges(self, subset):
        inner_vertices = select_subset(subset, self.n - 1)
        inner_edges = []

        v = len(inner_vertices)
        for i in range(v):
            for j in range(i + 1, v):
                x = inner_vertices[i]
                y = inner_vertices[j]
                if self.is_edge(x, y):
                    inner_edges.append((x, y, self.get_weight((x, y))))
                    assert inner_edges[-1][2] > 0

        return inner_edges

    def get_scores(self, subset):
        inner_edges = self.get_inner_edges(subset)

        return [
            self.gm(inner_edges),
            self.hm(inner_edges),
            self.am(inner_edges)
        ]

    def gm(self, inner_edges):
        if len(inner_edges) == 0:
            return 0
        res = 1.0
        for edge in inner_edges:
            w = edge[2]
            res *= pow(w, 1 / len(inner_edges))
        return res

    def hm(self, inner_edges):
        if len(inner_edges) == 0:
            return 0
        res = 0
        for edge in inner_edges:
            w = edge[2]
            res += 1 / w
        return len(inner_edges) / res

    def am(self, inner_edges):
        if len(inner_edges) == 0:
            return 0
        res = 0
        for edge in inner_edges:
            w = edge[2]
            res += w
        return res / len(inner_edges)

    def cn(self, inner_edges):


if __name__ == '__main__':
    n, paper_author_data, public_true_data, public_false_data, private_data = load_dataset(
    )

    n_order = 2

    if not os.path.exists(str(n_order) + "order"):
        os.mkdir(str(n_order) + "order")

    paper_author_data.sort(key=lambda x: len(x))
    public_true_data.sort(key=lambda x: len(x))
    public_false_data.sort(key=lambda x: len(x))

    h_edges = [set(authors) for authors in paper_author_data]
    exp = Expansion(n_order, h_edges)

    true_h_edges = [set(authors) for authors in public_true_data]
    false_h_edges = [set(authors) for authors in public_false_data]

    # true_scores = []
    # for i, h_edge in enumerate(true_h_edges):
    #     print(i + 1, "/", len(true_h_edges), "len:", len(h_edge))
    #     true_scores.append(exp.get_scores(h_edge))
    # false_scores = []
    # for i, h_edge in enumerate(false_h_edges):
    #     print(i + 1, "/", len(false_h_edges))
    #     false_scores.append(exp.get_scores(h_edge))

    true_scores = [exp.get_scores(h_edge) for h_edge in true_h_edges]
    false_scores = [exp.get_scores(h_edge) for h_edge in false_h_edges]

    gm_scores = []
    hm_scores = []
    am_scores = []
    for gm, hm, am in true_scores:
        gm_scores.append(gm)
        hm_scores.append(hm)
        am_scores.append(am)

    print("true")
    print(np.average(gm_scores))
    print(np.average(hm_scores))
    print(np.average(am_scores))

    x = np.arange(len(true_scores))
    fig = plt.figure(figsize=(7, 7))
    plt.plot(x, gm_scores)
    plt.savefig(str(n_order) + "order/true_gm")

    gm_scores.sort()
    fig = plt.figure(figsize=(7, 7))
    plt.plot(x[:15000], gm_scores[:15000])
    plt.savefig(str(n_order) + "order/true_gm_sort")

    fig = plt.figure(figsize=(7, 7))
    plt.plot(x, hm_scores)
    plt.savefig(str(n_order) + "order/true_hm")

    hm_scores.sort()
    fig = plt.figure(figsize=(7, 7))
    plt.plot(x[:15000], hm_scores[:15000])
    plt.savefig(str(n_order) + "order/true_hm_sort")

    fig = plt.figure(figsize=(7, 7))
    plt.plot(x, am_scores)
    plt.savefig(str(n_order) + "order/true_am")

    am_scores.sort()
    fig = plt.figure(figsize=(7, 7))
    plt.plot(x[:15000], am_scores[:15000])
    plt.savefig(str(n_order) + "order/true_am_sort")

    gm_ylim = max(gm_scores)
    gm_sort_ylim = max(gm_scores[:15000])
    hm_ylim = max(hm_scores)
    hm_sort_ylim = max(hm_scores[:15000])
    am_ylim = max(am_scores)
    am_sort_ylim = max(am_scores[:15000])

    gm_scores = []
    hm_scores = []
    am_scores = []
    for gm, hm, am in false_scores:
        gm_scores.append(gm)
        hm_scores.append(hm)
        am_scores.append(am)

    print("false")
    print(np.average(gm_scores))
    print(np.average(hm_scores))
    print(np.average(am_scores))

    x = np.arange(len(false_scores))
    fig = plt.figure(figsize=(7, 7))
    plt.plot(x, gm_scores)
    plt.ylim(top=gm_ylim)
    plt.savefig(str(n_order) + "order/false_gm")

    gm_scores.sort()
    fig = plt.figure(figsize=(7, 7))
    plt.plot(x[:15000], gm_scores[:15000])
    plt.ylim(top=gm_sort_ylim)
    plt.savefig(str(n_order) + "order/false_gm_sort")

    fig = plt.figure(figsize=(7, 7))
    plt.plot(x, hm_scores)
    plt.ylim(top=hm_ylim)
    plt.savefig(str(n_order) + "order/false_hm")

    hm_scores.sort()
    fig = plt.figure(figsize=(7, 7))
    plt.plot(x[:15000], hm_scores[:15000])
    plt.ylim(top=hm_sort_ylim)
    plt.savefig(str(n_order) + "order/false_hm_sort")

    fig = plt.figure(figsize=(7, 7))
    plt.plot(x, am_scores)
    plt.ylim(top=am_ylim)
    plt.savefig(str(n_order) + "order/false_am")

    am_scores.sort()
    fig = plt.figure(figsize=(7, 7))
    plt.plot(x[:15000], am_scores[:15000])
    plt.ylim(top=am_sort_ylim)
    plt.savefig(str(n_order) + "order/false_am_sort")

    # assert len(true_scores) == len(false_scores)

    # gm_scores = []
    # hm_scores = []
    # am_scores = []

    # x = np.arange(len(true_scores))

    # for ts, fs in zip(true_scores, false_scores):
    #     gm = ts[0] - fs[0]
    #     hm = ts[1] - fs[1]
    #     am = ts[2] - fs[2]
    #     gm_scores.append(gm)
    #     hm_scores.append(hm)
    #     am_scores.append(am)

    # fig = plt.figure(figsize=(7, 7))
    # plt.plot(x, gm_scores)
    # plt.savefig(str(n_order)+"order/gm")

    # gm_scores.sort()
    # fig = plt.figure(figsize=(7, 7))
    # plt.plot(x, gm_scores)
    # plt.savefig(str(n_order)+"order/gm_sort")

    # fig = plt.figure(figsize=(7, 7))
    # plt.plot(x, hm_scores)
    # plt.savefig(str(n_order)+"order/hm")

    # hm_scores.sort()
    # fig = plt.figure(figsize=(7, 7))
    # plt.plot(x, hm_scores)
    # plt.savefig(str(n_order)+"order/hm_sort")

    # fig = plt.figure(figsize=(7, 7))
    # plt.plot(x, am_scores)
    # plt.savefig(str(n_order)+"order/am")

    # am_scores.sort()
    # fig = plt.figure(figsize=(7, 7))
    # plt.plot(x, am_scores)
    # plt.savefig(str(n_order)+"order/am_sort")
