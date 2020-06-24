import random
import numpy as np
from numpy.linalg import norm

from util import get_two_random

random.seed(0)
np.random.seed(0)


class NMOEA:
    def __init__(self, graph, s_dict, d_dict, population_size=100, max_community_cnt=1000):
        self.graph = graph
        self.nodes = sorted(list(graph.nodes()))
        self.s_dict = s_dict
        self.d_dict = d_dict
        self.population_size = population_size
        self.max_community_cnt = max_community_cnt

    def run(self):
        """
        Run NMOEA algorithm based on NSGA-2
        """

        print("Run NMOEA algorithm!")

        # Random initialization of population
        population = []
        for _ in range(self.population_size):
            new_individual = self.nodes.copy()
            random.shuffle(new_individual)
            population.append(new_individual)
        update_count = self.population_size

        it = 1
        while update_count > 0:
            for i in range(self.population_size):
                # Except the first loop, population is sorted by fast-non-dominated-sort.
                # Select more desirable one from two random individuals.
                i = get_two_random(0, self.population_size - 1)[0]
                new_individual = self.reverse_operator(population[i])
                population.append(new_individual)

            population, update_count = self.fast_non_dominated_sort(population)

            print("Iter:{}, population is updated by {}".format(it, update_count))

    def reverse_operator(self, individual):
        """
        Individual is a permutation of all nodes
        Reverse a random segement in the given individual
        """
        assert len(individual) == len(self.nodes)

        i, j = get_two_random(0, len(individual) - 1)
        return individual[: i] + list(reversed(individual[i: j + 1])) + individual[j + 1:]

    def fast_non_dominated_sort(self, population):
        assert len(population) == 2 * self.population_size

        print("Fast non dominated sort")

        candidates = [self.fitness(individual) for individual in population]

        n = 2 * self.population_size
        dominate = [[] * n]
        counter = [0] * n
        rank = [-1] * n
        rank_store = [[]]

        print("A")
        for i, p in enumerate(candidates):
            for j, q in enumerate(candidates[i+1:]):
                if p != q and p[0] >= q[0] and p[1] >= q[1]:  # if p dominates q
                    dominate[i].append(j)
                    counter[j] += 1
                else:
                    dominate[j].append(i)
                    counter[i] += 1

            if counter[i] == 0:  # i must be a pareto-front
                rank[i] = 0
                rank_store[0].append(i)

        print("B")

        k = 0
        while len(rank_store[k]) > 0:
            assert len(rank_store) == k + 1
            rank_store.append([])
            for i in rank_store[k]:
                for j in dominate[i]:
                    counter[j] -= 1
                    if counter[j] == 0:
                        rank[j] = k + 1
                        rank_store[k + 1]. append(i)
            k += 1

        k, size = 0, 0
        new_population_idx = []
        while size < self.population_size:
            m = len(rank_store[k])
            crowding_distance = [0] * m

            rank_store[k].sort(key=lambda i: candidates[i][0])

            crowding_distance[0] = 1e9
            crowding_distance[m - 1] = 1e9

            f0_min = min([candidates[i][0] for i in rank_store[k]])
            f0_max = max([candidates[i][0] for i in rank_store[k]])
            f1_min = min([candidates[i][1] for i in rank_store[k]])
            f1_max = max([candidates[i][1] for i in rank_store[k]])

            scale0 = 1 / (f0_max - f0_min)
            scale1 = 1 / (f1_max - f1_min)

            for j in range(1, m - 1):
                i_m = rank_store[k][j - 1]
                i_p = rank_store[k][j + 1]

                d0 = (candidates[i_p][0] - candidates[i_m][0]) * scale0
                d1 = (candidates[i_m][1] - candidates[i_p][1]) * scale1
                crowding_distance[j] = d0 + d1

            new_population_idx += sorted(rank_store[k],
                                         key=lambda i: -crowding_distance[i])

            size += len(rank_store[k])
            k += 1

        new_population_idx = new_population_idx[:self.population_size]
        new_population = [population[i] for i in new_population_idx]

        update_count = 0
        for i in new_population_idx:
            if i >= self.population_size:
                update_count += 1

        return new_population, update_count

    def fitness(self, individual):
        """
        Return a tuple of two fitness values for the given individual
        """

        print("Start fitness")

        s_tot, d_tot = 0, 0
        communities = self.decode_communities(individual)

        print("Decode end")

        for i, community in enumerate(communities):
            print(i, "/", len(communities), "community size:", len(community))
            s_per_comm, d_per_comm = 0, 0
            m = len(community)

            if m == 1:  # 한개일때 fit VALUE 너무 커지는데 어떻게 할지??
                s_tot += 0.1
                d_tot += 0.1
                continue

            # for idx, node1 in enumerate(community):
            #     for node2 in community[idx + 1:]:
            sample_cnt = min(1000, m)
            for _ in range(sample_cnt):
                ii, jj = get_two_random(0, m - 1)
                node1, node2 = community[ii], community[jj]
                v1, v2 = self.s_dict[node1], self.s_dict[node2]
                s_per_comm += np.dot(v1, v2) / (norm(v1) * norm(v2))

                v1, v2 = self.d_dict[node1], self.d_dict[node2]
                d_per_comm += 1 - np.dot(v1, v2) / (norm(v1) * norm(v2))

            s_tot += s_per_comm / sample_cnt
            d_tot += d_per_comm / sample_cnt

        s_tot /= len(communities)
        d_tot /= len(communities)

        return (s_tot, d_tot)

    def decode_communities(self, individual, alpha=1):
        """
        Decode a permutation of nodes (individual) to list of communities
        """

        def community_fitness(int_deg, ext_deg):
            return int_deg / ((int_deg + ext_deg) ** alpha)

        print("Start decode")
        communities = []
        internal_degree = []
        external_degree = []
        node_to_community = {i: [] for i in self.nodes}

        for i, v in enumerate(individual):
            print(i, "/", len(individual), "community cnt:", len(communities))

            neighbors = list(self.graph.adj[v].keys())

            check = False

            community_candidates = []
            for neighbor in neighbors:
                community_candidates += node_to_community[neighbor]
            community_candidates = sorted(list(set(community_candidates)))

            # for j in range(len(communities)): # Too slow
            for j in community_candidates:
                int_deg_add, ext_deg_add = 0, 0
                for neighbor in neighbors:
                    if neighbor in communities[j]:
                        int_deg_add += 1
                        ext_deg_add -= 1
                    else:
                        ext_deg_add += 1

                f1 = community_fitness(internal_degree[j], external_degree[j])
                f2 = community_fitness(
                    internal_degree[j] + int_deg_add, external_degree[j] + ext_deg_add)

                if f1 < f2:
                    communities[j].add(v)
                    node_to_community[v].append(j)
                    internal_degree[j] += int_deg_add
                    external_degree[j] += ext_deg_add
                    check = True

            if not check:
                # if random.random() > 0.5:
                #     j = random.randint(0, len(communities) - 1)
                #     communities[j].add(v)

                #     for neighbor in neighbors:
                #         if neighbor in communities[j]:
                #             internal_degree[j] += 1
                #             ext_deg_add -= 1
                #         else:
                #             external_degree[j] += 1
                # else:
                communities.append(set([v]))
                node_to_community[v].append(len(communities) - 1)
                internal_degree.append(0)
                external_degree.append(len(neighbors))

        # Use hash table (dict) instead of list for fast deletion
        communities = {i: c for i, c in enumerate(communities)}

        print("community cnt:", len(communities))

        # while len(communities) > self.max_community_cnt:
        #     candidates = []
        #     for _ in range(10):
        #         i, j = get_two_random(0, len(communities) - 1)
        #         ci, cj = communities[i], communities[j]
        #         overlap = len(ci.intersection(cj)) / min(len(ci), len(cj))
        #         candidates.append((overlap, len(ci) + len(cj), i, j))

        #     _, _, i, j = max(candidates)
        #     communities[i] = communities[i].union(communities[j])
        #     del communities[j]

        return [list(c) for c in communities.values()]

    # def community_fitness(self, community, new_node=None, alpha=1):
    #     """
    #     """
    #     internal_degree, external_degree = 0, 0
    #     for edge in self.graph.edges(community):
    #         u, v = edge
    #         if u in community and v in community:
    #             internal_degree += 1
    #         else:
    #             external_degree += 1

    #     res1 = internal_degree / ((internal_degree + external_degree) ** alpha)

    #     if new_node is None or new_node in community:
    #         return res1

    #     for neighbor in self.graph.adj[new_node].keys():
    #         if neighbor in community:
    #             internal_degree += 1
    #             external_degree -= 1
    #         else:
    #             external_degree += 1

    #     res2 = internal_degree / ((internal_degree + external_degree) ** alpha)
    #     return res1, res2
