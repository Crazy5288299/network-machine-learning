import matplotlib.pyplot as plt
import networkx as nx
import math

# load the karate club graph
G = nx.read_edgelist('data/facebook_combined.txt', delimiter=' ', nodetype=int)

# Clustering
Clus = nx.average_clustering(G)
print("Clustering:", Clus)

if nx.is_connected(G):
    # Average distance
    ave_len = nx.average_shortest_path_length(G)
    print("Average distance:", ave_len)

    # Diameter
    Diameter = nx.diameter(G)
    print("Diameter:", Diameter)

# Degree
# 存储度数相应点数
number = []
# 存储度数
degree = []
for i in nx.degree_histogram(G):
    number.append(i)
for j in range(len(nx.degree_histogram(G))):
    degree.append(j)
# 去掉number=0,并取log
logxy = {}
for i in range(len(degree)):
    if (number[i] != 0):
        logxy[math.log(degree[i])] = math.log(number[i])

# 作图
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("degree distribution")
plt.xlabel("log(degree)")
plt.ylabel("log(number)")
plt.scatter(logxy.keys(), logxy.values(), c="red", s=10)
# plt.show()
plt.savefig("figure/random_degree")