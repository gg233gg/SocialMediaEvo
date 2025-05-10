import networkx as nx
import numpy as np

# 构建社交网络
def build_social_network(n_nodes=100, p=0.1):
    G = nx.erdos_renyi_graph(n=n_nodes, p=p)
    opinions = np.random.rand(n_nodes)
    return G, opinions

# 更新观点
def update_opinions(G, opinions, actions, step_size=0.1):
    n_nodes = len(opinions)
    new_opinions = opinions.copy()
    for i in range(n_nodes):
        if actions[i] == -1:
            neighbors = list(G.neighbors(i))
            if neighbors:
                avg_neighbor = np.mean([opinions[j] for j in neighbors])
                new_opinions[i] += (avg_neighbor - opinions[i]) * step_size
        elif actions[i] == +1:
            new_opinions[i] = 1.0 if opinions[i] >= 0.5 else 0.0
    return np.clip(new_opinions, 0, 1)