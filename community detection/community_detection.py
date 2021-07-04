import community
from community import community_louvain
import infomap
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx

def louvain(G, pos):
    # compute the best partition
    partition = community_louvain.best_partition(G)

    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=15,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

    modularity = community.modularity(partition, G)
    print([max(partition.values()) + 1, modularity])

    return max(partition.values()) + 1

def random_walk(G, pos):
    infomapWrapper = infomap.Infomap("--two-level --silent")
    for e in G.edges():
        infomapWrapper.addLink(*e)
    infomapWrapper.run()
    tree = infomapWrapper

    partition = {}
    for node in tree.iterTree():
        partition[node.physicalId] = node.moduleIndex()

    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=15,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

    modularity = community.modularity(partition, G)
    print([max(partition.values()) + 1, modularity])

    return tree.numTopModules()


# load the graph
# G = nx.karate_club_graph()
G = nx.read_edgelist('data/facebook_combined.txt', delimiter=' ', nodetype=int)
pos = nx.spring_layout(G)

louvain(G, pos)

random_walk(G, pos)