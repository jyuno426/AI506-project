import json
import random
import numpy as np
from tqdm import tqdm
from sklearn import svm
from numpy.linalg import norm
from matplotlib import pyplot as plt
from util import get_two_random, partition, means, standardize

random.seed(0)
np.random.seed(0)


class NMOEA:
    def __init__(self, node_num, graph, s_dict, d_dict, population_size=50, sample_cnt=100):
        self.node_num = node_num
        self.nodes = list(range(1, self.node_num + 1))
        self.graph = graph
        self.s_dict = s_dict
        self.d_dict = d_dict
        self.population_size = population_size
        self.sample_cnt = sample_cnt

    def run(self):
        """
        Run NMOEA algorithm based on NSGA-2
        """

        print("Run NMOEA algorithm!")

        # Random initialization of population
        population = []
        for _ in range(self.population_size):
            new_individual = list(range(self.node_num))
            for i in range(int(0.4 * self.node_num)):
                v = random.choice(self.nodes)

                if v in self.graph.adj:
                    for neighbor in self.graph.adj[v].keys():
                        new_individual[neighbor - 1] = new_individual[v - 1]

            population.append(new_individual)

        with open("../output/results/iter{}.json".format(0), "w") as f:
            json.dump(population, f)
        it = 1
        update_count = self.population_size

        while update_count > 0:
            for i in range(self.population_size // 2):
                # Except the first loop, population is sorted by fast-non-dominated-sort.
                # Select from first 1/3
                a, b = get_two_random(0, (self.population_size - 1) // 3)
                population.append(self.crossover(population[a], population[b]))
                population.append(self.mutation(population[a]))
                # print("Iter:{} updating: {}/{}".format(it,
                #                                        i + 1, self.population_size // 2))

            population, fitness_list, update_count = self.fast_non_dominated_sort(
                population)

            print("Iter:{}, population is updated by {}".format(it, update_count))

            community_lens = [len(self.decode_communities(indiv))
                              for indiv in population[:10]]

            _fitness = []
            for x in fitness_list[:10]:
                _fitness.append((round(x[0], 4), round(x[1], 4)))
            print(community_lens)
            print(_fitness)
            with open("../output/results/report.txt", "a+") as f:
                f.write("Iter:{}, population is updated by {}\n".format(
                    it, update_count))
                f.write(str(community_lens) + "\n")
                f.write(str(_fitness) + "\n")
            with open("../output/results/iter{}.json".format(it), "w") as f:
                json.dump(population, f)
            it += 1

    def crossover(self, indiv1, indiv2):
        res = indiv2.copy()
        v = random.choice(self.nodes)
        for node in self.nodes:
            if indiv1[node - 1] == indiv1[v - 1]:
                res[node - 1] = indiv1[v - 1]
        return res

    def mutation(self, indiv):
        res = indiv.copy()
        u, v = get_two_random(1, len(self.nodes), ordering=False)
        for node in self.nodes:
            if res[node - 1] == indiv[v - 1]:
                res[node - 1] = indiv[u - 1]
        return res

    def fast_non_dominated_sort(self, population):
        assert len(population) == 2 * self.population_size

        # print("Fast non dominated sort")

        candidates = []
        with tqdm(desc="Compute fitness", total=len(population)) as pbar:
            for indiv in population:
                candidates.append(self.fitness(indiv))
                pbar.update(1)

        n = 2 * self.population_size
        dominate = [[] for _ in range(n)]
        counter = [0] * n
        rank = [-1] * n
        rank_store = [[]]

        for i, p in enumerate(candidates):
            for jj, q in enumerate(candidates[i+1:]):
                j = jj + i + 1
                if p == q:
                    continue
                if p[0] >= q[0] and p[1] >= q[1]:  # if p dominates q
                    dominate[i].append(j)
                    counter[j] += 1
                elif p[0] <= q[0] and p[1] <= q[1]:
                    dominate[j].append(i)
                    counter[i] += 1

            if counter[i] == 0:  # i must be a pareto-front
                rank[i] = 0
                rank_store[0].append(i)

        # print(dominate)
        # print(counter)
        # print(rank)
        # print(rank_store)

        k = 0
        while len(rank_store[k]) > 0:
            assert len(rank_store) == k + 1
            rank_store.append([])
            for i in rank_store[k]:
                for j in dominate[i]:
                    counter[j] -= 1
                    if counter[j] == 0:
                        rank[j] = k + 1
                        rank_store[k + 1].append(j)
            k += 1

        k, size = 0, 0
        new_population_idx = []
        while size < self.population_size:
            m = len(rank_store[k])
            # print(k, m)
            crowding_distance = [0] * m

            rank_store[k].sort(key=lambda i: candidates[i][0])

            crowding_distance[0] = 1e9
            crowding_distance[m - 1] = 1e9

            f0_min = min([candidates[i][0] for i in rank_store[k]])
            f0_max = max([candidates[i][0] for i in rank_store[k]])
            f1_min = min([candidates[i][1] for i in rank_store[k]])
            f1_max = max([candidates[i][1] for i in rank_store[k]])

            if f0_max - f0_min < 1e-9:
                scale0 = 1
            else:
                scale0 = 1 / (f0_max - f0_min)

            if f1_max - f1_min < 1e-9:
                scale1 = 1
            else:
                scale1 = 1 / (f1_max - f1_min)

            for j in range(1, m - 1):
                i_m = rank_store[k][j - 1]
                i_p = rank_store[k][j + 1]

                d0 = (candidates[i_p][0] - candidates[i_m][0]) * scale0
                d1 = (candidates[i_m][1] - candidates[i_p][1]) * scale1
                crowding_distance[j] = d0 + d1

            idx_to_idx = {idx: i for i, idx in enumerate(rank_store[k])}
            new_population_idx += sorted(rank_store[k],
                                         key=lambda idx: -crowding_distance[idx_to_idx[idx]])

            size += len(rank_store[k])
            k += 1

        new_population_idx = new_population_idx[:self.population_size]
        new_population = [population[i] for i in new_population_idx]
        fitness_list = [candidates[i] for i in new_population_idx]

        update_count = 0
        for i in new_population_idx:
            if i >= self.population_size:
                update_count += 1

        return new_population, fitness_list, update_count

    def fitness_community(self, community, random_sample=True):
        m = len(community)
        if m == 1:  # 한개일때 fit VALUE 너무 커지는데 어떻게 할지??
            return 0.1, 0.1

        def fitness_pair(node1, node2):
            if node1 not in self.s_dict:
                return 0, 0
            if node2 not in self.s_dict:
                return 0, 0
            if node1 not in self.d_dict:
                return 0, 0
            if node2 not in self.d_dict:
                return 0, 0

            v1, v2 = self.s_dict[node1], self.s_dict[node2]
            s = np.dot(v1, v2) / (norm(v1) * norm(v2))

            v1, v2 = self.d_dict[node1], self.d_dict[node2]
            d = 1 - np.dot(v1, v2) / (norm(v1) * norm(v2))

            return s, d

        s_tot, d_tot = 0, 0
        _community = list(community)

        if random_sample:
            sample_cnt = min(self.sample_cnt, m)
            for _ in range(sample_cnt):
                ii, jj = get_two_random(0, m - 1)
                node1, node2 = _community[ii], _community[jj]
                ss, dd = fitness_pair(node1, node2)
                s_tot += ss
                d_tot + dd
            s_tot /= sample_cnt
            d_tot /= sample_cnt
        else:
            for idx, node1 in enumerate(community):
                for node2 in community[idx + 1:]:
                    ss, dd = fitness_pair(node1, node2)
                    s_tot += ss
                    d_tot + dd
            n = len(community) * (len(community) - 1) / 2
            s_tot /= n
            d_tot /= n

        return s_tot, d_tot

    def fitness(self, individual):
        """
        Return a tuple of two fitness values for the given individual
        """

        s_tot, d_tot = 0, 0
        communities = self.decode_communities(individual)

        for i, community in enumerate(communities):
            # print(i, "/", len(communities), "community size:", len(community))
            s_per_comm, d_per_comm = 0, 0
            m = len(community)

            if m == 1:  # 한개일때 fit VALUE 너무 커지는데 어떻게 할지??
                s_tot += 0.1
                d_tot += 0.1
                continue

            # for idx, node1 in enumerate(community):
            #     for node2 in community[idx + 1:]:
            sample_cnt = min(self.sample_cnt, m)
            for _ in range(sample_cnt):
                ii, jj = get_two_random(0, m - 1)
                node1, node2 = community[ii], community[jj]
                if node1 not in self.s_dict or node2 not in self.s_dict or node1 not in self.d_dict or node2 not in self.d_dict:
                    continue

                v1, v2 = self.s_dict[node1], self.s_dict[node2]
                s_per_comm += np.dot(v1, v2) / (norm(v1) * norm(v2))

                v1, v2 = self.d_dict[node1], self.d_dict[node2]
                d_per_comm += 1 - np.dot(v1, v2) / (norm(v1) * norm(v2))

            s_tot += s_per_comm / sample_cnt
            d_tot += d_per_comm / sample_cnt

        s_tot /= len(communities)
        d_tot /= len(communities)

        return (s_tot, d_tot)

    def decode_communities(self, individual):
        community_table = {}
        for node in self.nodes:
            c = individual[node - 1]
            if c in community_table:
                community_table[c].add(node)
            else:
                community_table[c] = set([node])

        return [list(c) for c in community_table.values()]

    def gather_pareto_front_communities(self, population):
        candidates = []
        for indiv in tqdm(population, desc="Compute fitness"):
            evaluation = self.fitness(indiv)
            candidates.append([evaluation, self.decode_communities(indiv)])

        pareto_front = []
        candidates.sort()
        for evaluation, community in tqdm(candidates, desc="Get pareto front"):
            x, y = evaluation
            while len(pareto_front) > 0:
                if pareto_front[-1][0][0] <= x and pareto_front[-1][0][1] <= y:
                    pareto_front.pop()
                else:
                    break
            pareto_front.append([evaluation, community])

        communities = []
        for front in tqdm(pareto_front, desc="Gather communities"):
            for community in front[1]:
                if len(community) >= 2:
                    communities.append(set(community))

        return communities

    def get_n_in_out(self, population, coauthor_list):
        # communities = [set(c) for c in self.decode_communities(
        #     random.choice(population))]

        communities = self.gather_pareto_front_communities(population)

        res = []
        for authors in tqdm(coauthor_list, desc="Compute features: n_in, n_out"):
            n_in = 0
            n_out = 0
            for i, author1 in enumerate(authors):
                for author2 in authors[i + 1:]:
                    check = False
                    for community in communities:
                        if author1 in community and author2 in community:
                            n_in += 1
                            check = True
                    if not check:
                        n_out += 1

            res.append((n_in, n_out))

        return np.array(res)

    def get_features(self, population, coauthor_list):
        """
        """
        communities = self.gather_pareto_front_communities(population)

        res = []
        for authors in tqdm(coauthor_list, desc="Compute features"):
            comm_len_1 = []
            comm_len_2 = []
            # comm_fit_s_1 = []
            # comm_fit_d_1 = []
            # comm_fit_s_2 = []
            # comm_fit_d_2 = []
            for i, author1 in enumerate(authors):
                for community in communities:
                    if author1 in community:
                        s, d = self.fitness_community(community)
                        # comm_fit_s_1.append(s)
                        # comm_fit_d_1.append(d)
                        comm_len_1.append(len(community))
                        for author2 in authors[i + 1:]:
                            if author2 in community:
                                # comm_fit_s_2.append(s)
                                # comm_fit_d_2.append(d)
                                comm_len_2.append(len(community))

            mean_1 = means(comm_len_1)
            mean_2 = means(comm_len_2)
            # mean_fit_s_1 = means(comm_fit_s_1)
            # mean_fit_d_1 = means(comm_fit_d_1)
            # mean_fit_s_2 = means(comm_fit_s_2)
            # mean_fit_d_2 = means(comm_fit_d_2)

            features = [len(authors), len(comm_len_1),
                        len(comm_len_2)] + mean_1 + mean_2
            # + mean_fit_s_1 + mean_fit_d_1 + mean_fit_s_2 + mean_fit_d_2

            res.append(features)

        return np.array(standardize(res))

    def fit(self, population, coauthor_list, labels):
        # plt.figure(figsize=(7, 7))
        # plt.plot(true_n_in, true_n_out, "bo")
        # plt.plot(false_n_in, false_n_out, "ro")
        # plt.savefig("../output/plot_1_indiv_random.png")

        X = self.get_features(population, coauthor_list)
        y = np.array(labels)

        print("Fitting svm ...")
        self.predictor = svm.SVC()
        self.predictor.fit(X, y)
        return self.predictor.predict(X)

    def eval(self, population, coauthor_list):
        X = self.get_features(population, coauthor_list)
        return self.predictor.predict(X)
