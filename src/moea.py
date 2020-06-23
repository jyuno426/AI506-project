import random
import copy
import numpy as np
import json
import pickle



def remove_subset(children):
    remove_list = []
    for idx, c1 in enumerate(children):
        for c2 in children[idx+1:]:
            if (set(c1) <= set(c2)):
                remove_list.append(c1)
            elif (set(c1)>= set(c2)):
                remove_list.append(c2)
    children = [e for e in children if e not in remove_list]
    return children

def crossover(child):
    # pick two random clusters
    c1, c2 = random.sample(child, 2) 
    child.remove(c1)
    child.remove(c2)

    #remove if they have an intersection
    inter = set(c1).intersection(set(c2))
    set_c1 = set(c1) - inter
    set_c2 = set(c2) - inter
    

    # 나중에 연결된 부분만 고르도록 해도 될듯
    c1_len = random.randint(1, len(set_c1))
    c2_len = random.randint(1, len(set_c2))

    new_c1 = set(random.sample(set_c1, c1_len))
    new_c2 = set(random.sample(set_c2, c2_len))
    
    left_c1 = set_c1 - new_c1
    left_c2 = set_c2 - new_c2

    # make new crossovered cluster
    left_c1.update(new_c2, inter)
    left_c2.update(new_c1, inter)

    child.append(list(left_c1))
    child.append((list(left_c2)))
    child = remove_subset(child)
    #print("after remove", child)
    return child

def mutation(child):
    new_children = []
    ## 1. make new cluster ##
    c1 = random.sample(child, 1)[0]
    child1 = copy.deepcopy(child)
    #print(c1)
    

    # pick some nodes
    if len(c1)!= 1:
        child1.remove(c1)
        c1_len = random.randint(1, len(c1)-1)
        new = random.sample(c1, c1_len)
        new2 = list(set(c1) - set(new))

        child1.append(new)
        child1.append(new2)
        child1 = remove_subset(child1)
        new_children.append(child1)
        #print("1.",child1)
        


    ## 2. remove and add to other cluster ##
    child2 = copy.deepcopy(child)
    c1, c2 = random.sample(child2, 2)
    
    if len(c1)!= 1:
        child2.remove(c1)
        child2.remove(c2)
        c1_len = random.randint(1, len(c1)-1)
        new = random.sample(c1, c1_len)
        c1 =list(set(c1) - set(new))
        c2 = list(set(c2).union(new))
        child2+= [c1, c2]
        child2 = remove_subset(child2)
        new_children.append(child2)
        #print("2",child2)

    ## 3. merge ##
    child3 = copy.deepcopy(child)
    c1, c2 = random.sample(child3, 2)
    c3 = list(set(c1).union(set(c2)))
    child3.remove(c1)
    child3.remove(c2)
    child3.append(c3)
    child3 = remove_subset(child3)
    new_children.append(child3)
    #print("3",child3)


    return new_children

def fit_eval(child):
    global s_dict, d_dict, sum_of_all_d, adj_mat, all_nodes
    s_tot = 0
    d_tot = 0
    #print("new child--")
    for cluster in child:
        s_per_cluster = 0
        if len(cluster)==1: # 한개일때 fit VALUE 너무 커지는데 어떻게 할지??
            s_per_cluster =1
            s_tot += s_per_cluster
            continue
        
        for idx, node1 in enumerate(cluster):
            for node2 in cluster[idx+1:]:
                s_per_cluster += np.dot(s_dict[node1], s_dict[node2]) / (np.linalg.norm(s_dict[node1]) * np.linalg.norm(s_dict[node2]))
        
        s_per_cluster = s_per_cluster/( len(cluster)* (len(cluster) -1) ) 
        s_tot += s_per_cluster

    for cluster in child:
        d_per_cluster = 0
        if len(cluster)==1: # 한개일때 fit VALUE 너무 커지는데 어떻게 할지??
            d_per_cluster =0
            d_tot += d_per_cluster
            continue
        
        for idx, node1 in enumerate(cluster):
            for node2 in cluster[idx+1:]:
                ## 여기 맞는지 물어보고 다시짜기!!
                d_per_cluster -= 1-( np.dot(d_dict[node1], d_dict[node2]) / (np.linalg.norm(d_dict[node1]) * np.linalg.norm(d_dict[node2])))
        
        # have to calculate the number of edges !!
        l = 0
        for idx, node1 in enumerate(cluster):
            for node2 in all_nodes: 
                if node2 not in cluster[idx+1:] :
                    l += adj_mat[int(node1)-1, int(node2)-1]
        d_tot += d_per_cluster

    d_tot = (d_tot + sum_of_all_d )/l
    # need to savfe s_dict, d_dict here!
    
    return [s_tot, d_tot]

def pareto_front(popul):
    # one of popul  = list of communities
    # return the pareto_front (list) which are subset of population
    # fit_eval(one of population)  = [s_fit, d_fit]

    max_fits = []
    fit_dict = {}
    for p in popul:
        val = fit_eval(p)
        fit_dict[tuple(val)] = p
        max_fits.append(val)

    max_fits.sort()
    res= []

    for x,y in max_fits:
        while len(res) >0:
            if res[-1][0] <= x and res[-1][1]<=y:
                res.pop()
            else:
                break
        res.append((x,y))

    '''

    x = [f[0] for f,p in res]
    y = [f[1] for f,p in res]
    import matplotlib.pyplot as plt
    plt.scatter(x,y)
    plt.show()
    '''
    print("pareto front value pairs:",[f for f in res])
    new_popul = [fit_dict[k] for k in res]
    
    return new_popul

if __name__ == '__main__':   

    MaxCount = 1

    # dictionary 에 노드 넘버: vector값 저장
    s_dict = {}
    d_dict = {}
    with open('../output/node_embedding_p1_q0.5.json') as json_1:
        s_dict = json.load(json_1)

    with open('../output/node_embedding_p1_q2.json') as json_1:
        d_dict = json.load(json_1)

    keys = list(s_dict.keys())
    for k in keys:
        s_dict[int(k)] = s_dict[k]
        del s_dict[k]

    keys = list(d_dict.keys())
    for k in keys:
        d_dict[int(k)] = d_dict[k]
        del d_dict[k]

    # used for caculating D
    sum_of_all_d = 0
    all_nodes = list(d_dict.keys())

    for idx, node1 in enumerate(all_nodes):
        for node2 in all_nodes[idx+1:]:
            sum_of_all_d += 1-( np.dot(d_dict[node1], d_dict[node2]) / (np.linalg.norm(d_dict[node1]) * np.linalg.norm(d_dict[node2])))

    # adj_mat[i,j] <- edge between node i+1 and node j+1
    adj_mat = np.load("../output/adj_mat.npy")

    
    population=[]

    ### fill the line 5,6 of given pseudo code. #### modify child!!
    #Child = [[1,2],[2,3,4],[5]
    child = np.load("../output/co_author.npy", allow_pickle=True) 
    child_list = []
    for i in range(len(child)):
        child_list.append(child[i])
    population.append(child_list)

    for count in range(0, MaxCount): 
        picked_child = random.choice(population) # it does not remove original child.

        crossover_child = crossover(picked_child)
        population.append(crossover_child)
        mutation_children = []
        for c in population:
            mutation_children += mutation(c)
        
        population += mutation_children


        print("Iter: {}, # of popul:{}".format(count, len(population)))
        population = pareto_front(population)
        print("# of pf:{}".format( len(population)))
        print("------------------------")
        
    # save pareto front to txt file
    g = open("../output/fin_popul_{}_iter.txt".format(MaxCount), 'w')

    for i, p in enumerate(population):
        g.write('p'+str(i) +'\n')
        for c in p:
            g.write(" ".join(list(map(str, c)))+'\n')
    g.close()
    
    check = []
    for p in population:
        for clusters in p:
            check.append(set(clusters))
    
    # for test
    f = open("../output/test.txt", 'r')
    lines = f.readlines()
    for line in lines:
        cnt = 0
        inp = set(map(int, line.strip().split(' ')))
        for cluster in check:
            if inp <= cluster:
                cnt +=1
        print(cnt/len(check))

    f.close()


