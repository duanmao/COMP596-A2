import collections
import numpy as np
import pickle as pkl
import community
import networkx as nx
import networkx.algorithms.community as nxcom
import scipy.sparse as sp
import sklearn.metrics as sm
# from scipy.sparse.linalg.eigen.arpack import eigsh
import os
import re
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
from igraph import *
import random
from gcn import utils
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

DEBUG = False
img_format = "png"

# dataset: real-classic, real-node-label (gcn), synthetic (nx LFR)
classic_path = "datasets/real-classic/"
classic_files = ["polbooks", "polblogs", "karate", "football", "strike"]

# gcn_filepath = "datasets/gcn"
gcn_files = ["citeseer", "cora", "pubmed"]

def randomColor():
    return "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

def partitionListToMembership(partList):
    memdict = {}
    for i, com in enumerate(partList):
        for v in com:
            memdict[v] = i
    return [memdict[v] for v in sorted(memdict)]

def toIGraph(nxgraph):
    return Graph(len(nxgraph), list(zip(*list(zip(*nx.to_edgelist(nxgraph)))[:2])))

# load real-classic
def loadRealClassic(dataset):
    fname = os.path.join(classic_path, f"{dataset}.gml")
    nxgraph = nx.read_gml(fname, label=None)
    iggraph = Graph.Read_GML(fname)
    print(f'\t {len(nxgraph)} nodes')
    if DEBUG: print(len(nxgraph), iggraph)

    if dataset == 'karate':
        y = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        y = [data["value"] for node, data in nxgraph.nodes(data = True)]
        if not isinstance(y[0], int):
            mapping = {}
            for i, c in enumerate(y):
                if c not in mapping: mapping[c] = len(mapping)
                y[i] = mapping[c]
    if DEBUG: print(y)
    
    gtcount = max(y) + 1
    print(f'\t Ground Truth: {gtcount} communities')
    return nxgraph, iggraph, y, gtcount

# load gcn data
def loadGcnData(dataset):
    nxgraph, y_mtx = utils.load_data(dataset)
    print(f'\t\t {len(nxgraph)} nodes')
    
    if DEBUG:
        print("Graph", nxgraph, len(nxgraph))
        print("y", np.shape(y_mtx))

    y = np.argmax(y_mtx, axis=1)
    if DEBUG:
        print(y_mtx, y)

    gtcount = y_mtx.shape[1]
    print(f'\t\t Ground Truth: {gtcount} communities')
    return nxgraph, toIGraph(nxgraph), y, gtcount

# load networkx LFR
def loadLFR(ith):
    n = [200, 200, 211, 217, 222]
    tau1 = 2.5
    tau2 = 1.5
    mu = 0.1
    mindegree = [5, 5, 5, 3, 6]
    seed = [10, 11, 10, 11, 10]
    graph = nxcom.LFR_benchmark_graph(n[ith], tau1, tau2, mu, min_degree=mindegree[ith], min_community=10, seed=seed[ith])
    print(f'\t {n[ith]} nodes')

    communities = {frozenset(graph.nodes[v]['community']) for v in graph} # partition list
    gtcount = len(communities)
    print(f'\t Ground Truth: {gtcount} communities')

    y = partitionListToMembership(communities)
    if DEBUG: print(communities, y)

    return graph, toIGraph(graph), y, gtcount

def drawClustered(nxgraph, communities, title):
    plt.clf()
    pos = nx.spring_layout(nxgraph)
    for nodes in communities:
        nx.draw_networkx_nodes(nxgraph, pos, nodes, node_size = 20,
                                    node_color = randomColor())
    nx.draw_networkx_edges(nxgraph, pos, alpha=0.5)
    plt.title(title)
    plt.savefig(f'{title}.{img_format}')

def evaluate(communities, nxgraph, iggraph, groundtruth, NMIs, ARIs, Qs, method):
    membership = partitionListToMembership(communities)
    if DEBUG: print(f'membership{membership}')
    # modularity
    Q = nxcom.modularity(nxgraph, communities)
    Q1 = iggraph.modularity(membership)
    if abs(Q - Q1) > 1e-5: print(f'\t\t Q: networkx.alg: {Q} \t igraph: {Q1}')

    # NMI
    NMI = sm.normalized_mutual_info_score(groundtruth, membership, average_method="arithmetic")
    NMI1 = compare_communities(membership, groundtruth, method="nmi")
    if abs(NMI - NMI1) > 1e-3: print(f'\t\t NMI \t sklearn: {NMI} \t igraph: {NMI1}')

    # ARI
    ARI = sm.adjusted_rand_score(groundtruth, membership)
    ARI1 = compare_communities(membership, groundtruth, method="ari")
    if abs(ARI - ARI1) > 1e-5: print(f'\t\t ARI \t sklearn: {ARI} \t igraph: {ARI1}')

    NMIs[method].append(NMI)
    ARIs[method].append(ARI)
    Qs[method].append(Q)
    return NMI, ARI, Q

def membershipToPartitionList(nxgraph, membership):
    start = min(nxgraph.node)
    partitions = collections.defaultdict(list)
    for node, label in enumerate(membership):
        partitions[label].append(node + start)
    return partitions.values()

def partition(nxgraph, iggraph, title, NMIs, ARIs, Qs, gtcount=None):
    def toParitionList(communityDict):
        partList = []
        for com in set(communityDict.values()) :
            partList.append([node for node in communityDict if communityDict[node] == com])
        return partList

    def fix_dendrogram(graph, cl):
        already_merged = set()
        for merge in cl.merges:
            already_merged.update(merge)

        num_dendrogram_nodes = graph.vcount() + len(cl.merges)
        not_merged_yet = sorted(set(range(num_dendrogram_nodes)) - already_merged)
        if len(not_merged_yet) < 2:
            return

        v1, v2 = not_merged_yet[:2]
        cl._merges.append((v1, v2))
        del not_merged_yet[:2]

        missing_nodes = range(num_dendrogram_nodes,
                num_dendrogram_nodes + len(not_merged_yet))
        cl._merges.extend(zip(not_merged_yet, missing_nodes))
        cl._nmerges = graph.vcount()-1

    if nxgraph.is_directed(): nxgraph = nxgraph.to_undirected()
    if iggraph.is_directed(): iggraph.to_undirected()

    # Louvain
    method = "Louvain"
    print(f'\t\t {method}:')

    comm_louv = community.best_partition(nxgraph) # a dict
    if DEBUG: print(comm_louv)
    print(f'\t\t {len(set(comm_louv.values()))} communities')

    comm_louv = toParitionList(comm_louv) # to partition last
    NMI, ARI, Q = evaluate(comm_louv, nxgraph, iggraph, y, NMIs, ARIs, Qs, method)
    drawClustered(nxgraph, comm_louv, f"{title} {method}")
    print(f'\t\t\t NMI: {NMI} \t ARI: {ARI} \t Modularity: {Q}')

    # fast modularity
    method = "Fast Modularity"
    print(f'\t\t {method}:')

    comm_fastMod = nxcom.greedy_modularity_communities(nxgraph) # a partition list
    if DEBUG: print(comm_fastMod)
    print(f'\t\t {len(comm_fastMod)} communities')

    NMI, ARI, Q = evaluate(comm_fastMod, nxgraph, iggraph, y, NMIs, ARIs, Qs, method)
    drawClustered(nxgraph, comm_fastMod, f"{title} {method}")
    print(f'\t\t\t NMI: {NMI} \t ARI: {ARI} \t Modularity: {Q}')

    # infomap
    method = "Infomap"
    print(f'\t\t {method}:')

    comm_infom = iggraph.community_infomap().membership # memberships
    if DEBUG: print(comm_infom)
    infomList = membershipToPartitionList(nxgraph, comm_infom)
    print(f'\t\t {max(comm_infom) + 1} communities')

    NMI, ARI, Q = evaluate(infomList, nxgraph, iggraph, y, NMIs, ARIs, Qs, method)
    drawClustered(nxgraph, infomList, f"{title} {method}")
    print(f'\t\t\t NMI: {NMI} \t ARI: {ARI} \t Modularity: {Q}')

    # walk trap
    method = "Walk Trap"
    print(f'\t\t {method}:')

    comm_wt = iggraph.community_walktrap().as_clustering().membership # memberships
    if DEBUG: print(comm_wt)
    wtList = membershipToPartitionList(nxgraph, comm_wt)
    print(f'\t\t {max(comm_wt) + 1} communities')

    NMI, ARI, Q = evaluate(wtList, nxgraph, iggraph, y, NMIs, ARIs, Qs, method)
    drawClustered(nxgraph, wtList, f"{title} {method}")
    print(f'\t\t\t NMI: {NMI} \t ARI: {ARI} \t Modularity: {Q}')

    # community_label_propagation
    method = "Label Propagation"
    print(f'\t\t {method}:')

    comm_lp = iggraph.community_label_propagation().membership # memberships
    if DEBUG: print(comm_lp)
    lpList = membershipToPartitionList(nxgraph, comm_lp)
    print(f'\t\t {max(comm_lp) + 1} communities')

    NMI, ARI, Q = evaluate(lpList, nxgraph, iggraph, y, NMIs, ARIs, Qs, method)
    drawClustered(nxgraph, lpList, f"{title} {method}")
    print(f'\t\t\t NMI: {NMI} \t ARI: {ARI} \t Modularity: {Q}')

    # community_leading_eigenvector
    method = "Leading Eigenvector"
    print(f'\t\t {method}:')

    comm_le = iggraph.community_leading_eigenvector().membership # memberships
    if DEBUG: print(comm_le)
    leList = membershipToPartitionList(nxgraph, comm_le)
    print(f'\t\t {max(comm_le) + 1} communities')

    NMI, ARI, Q = evaluate(leList, nxgraph, iggraph, y, NMIs, ARIs, Qs, method)
    drawClustered(nxgraph, leList, f"{title} {method}")
    print(f'\t\t\t NMI: {NMI} \t ARI: {ARI} \t Modularity: {Q}')

    if title.split()[1] not in {'polblogs', 'pubmed'}:
        # Edge Betweenness
        method = "Edge Betweenness"
        print(f'\t\t {method}:')

        comm_eb = iggraph.community_edge_betweenness()
        if not nx.is_connected(nxgraph):
            fix_dendrogram(iggraph, comm_eb)
        comm_eb = comm_eb.as_clustering().membership # memberships
        if DEBUG: print(comm_eb)
        ebList = membershipToPartitionList(nxgraph, comm_eb)
        print(f'\t\t {max(comm_eb) + 1} communities')

        NMI, ARI, Q = evaluate(ebList, nxgraph, iggraph, y, NMIs, ARIs, Qs, method)
        drawClustered(nxgraph, ebList, f"{title} {method}")
        print(f'\t\t\t NMI: {NMI} \t ARI: {ARI} \t Modularity: {Q}')

    try:
        # spinglass
        method = "Spinglass"
        print(f'\t\t {method}:')

        comm_sp = iggraph.community_spinglass().membership # memberships
        if DEBUG: print(comm_sp)
        spList = membershipToPartitionList(nxgraph, comm_sp)
        print(f'\t\t {max(comm_sp) + 1} communities')

        NMI, ARI, Q = evaluate(spList, nxgraph, iggraph, y, NMIs, ARIs, Qs, method)
        drawClustered(nxgraph, spList, f"{title} {method}")
        print(f'\t\t\t NMI: {NMI} \t ARI: {ARI} \t Modularity: {Q}')
    except:
        print("error")
        pass

    # if gtcount != None:
    #     print('\t Leading Eigenvector with Known Clusters:')

    #     comm_le = iggraph.community_leading_eigenvector(clusters=gtcount).membership # memberships
    #     if DEBUG: print(comm_le)
    #     leList = membershipToPartitionList(nxgraph, comm_le)
    #     print(f'\t {max(comm_le) + 1} communities')

    #     NMI, ARI, Q = evaluate(leList, nxgraph, iggraph, y)
    #     drawClustered(nxgraph, leList, f"{title} Leading Eigenvector with Known Clusters")
    #     print(f'\t\t NMI: {NMI} \t ARI: {ARI} \t Modularity: {Q}')

    # community_optimal_modularity
    # print('\t Optimal Modularity:')

    # ommember = iggraph.community_optimal_modularity().membership # memberships
    # if DEBUG: print(ommember)
    # omList = membershipToPartitionList(nxgraph, ommember)
    # print(f'\t {max(ommember) + 1} communities')

    # NMI, ARI, Q = evaluate(omList, nxgraph, iggraph, y)
    # drawClustered(nxgraph, omList, f"{title} Optimal Modularity")
    # print(f'\t\t NMI: {NMI} \t ARI: {ARI} \t Modularity: {Q}')

def average(NMIs, ARIs, Qs, allNMIs = None, allARIs = None, allQs = None):
    if DEBUG: print(f'\t {NMIs} \n\t {ARIs} \n\t {Qs}')
    print(f'\n\t Averages:')
    arr = [NMIs, ARIs, Qs]
    evls = ["NMI", "ARI", "Modularity"]
    alls = [allNMIs, allARIs, allQs]
    for evl, dataDict, allcontainer in zip(evls, arr, alls):
        print(f'\t  {evl}')
        means = {}
        for method in dataDict:
            if allcontainer != None: allcontainer[method].extend(dataDict[method])
            m = np.mean(dataDict[method])
            means[method] = m
            print(f'\t\t {method}: {m}')
        if DEBUG: print(means)
        winner = sorted([(means[method], method) for method in means])[-1][1]
        print(f'\t winner: {winner}')
    if DEBUG:
        print(f'\t {allNMIs} \n\t {allARIs} \n\t {allQs}')
    
allNMIs, allARIs, allQs = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)

# print('real-classic... \n')
# NMIs, ARIs, Qs = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
# for dataset in classic_files:
#     print(f'\t{dataset}')
#     nxgraph, iggraph, y, gtcount = loadRealClassic(dataset)
#     partition(nxgraph, iggraph, f"real-classic {dataset}", NMIs, ARIs, Qs, gtcount)
#     if DEBUG: break
# average(NMIs, ARIs, Qs, allNMIs, allARIs, allQs)

# print('GCN (real-node-label...) \n')
# NMIs, ARIs, Qs = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
# for dataset in gcn_files:
#     print(f'\t{dataset}')
#     nxgraph, iggraph, y, gtcount = loadGcnData(dataset)
#     partition(nxgraph, iggraph, f"real-node-label {dataset}", NMIs, ARIs, Qs, gtcount)
#     if DEBUG: break
# average(NMIs, ARIs, Qs, allNMIs, allARIs, allQs)

print('LFR (synthetic)... \n')
NMIs, ARIs, Qs = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
for i in range(5):
    nxgraph, iggraph, y, gtcount = loadLFR(i)
    partition(nxgraph, iggraph, f"LFR Benchmark (synthetic) {i}", NMIs, ARIs, Qs, gtcount)
average(NMIs, ARIs, Qs, allNMIs, allARIs, allQs)

average(allNMIs, allARIs, allQs)