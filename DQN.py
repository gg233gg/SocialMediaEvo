import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from torch_geometric.data import Data

from env import build_social_network, update_opinions
from model import DQNAgent, GATEncoder

# 超参数
n_nodes = 100
state_dim = 8
action_dim = 3
lr = 1e-3
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
batch_size = 32
memory_size = 10000

# 初始化网络
G, opinions = build_social_network(n_nodes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GAT 数据转换
def get_gat_data(opinions):
    x = torch.tensor(opinions, dtype=torch.float32).view(-1, 1)
    edges = np.array(G.edges).T
    edge_index = torch.tensor(edges, dtype=torch.long)
    return Data(x=x, edge_index=edge_index).to(device)

# 经验回放
memory = deque(maxlen=memory_size)

# 初始化智能体
gat_encoder = GATEncoder(input_dim=1, hidden_dim=state_dim//4).to(device)
actual_state_dim = gat_encoder.out_dim
agents = [DQNAgent(state_dim=actual_state_dim, action_dim=action_dim).to(device) for _ in range(n_nodes)]
targets = [DQNAgent(state_dim=actual_state_dim, action_dim=action_dim).to(device) for _ in range(n_nodes)]
optimizers = [optim.Adam(agents[i].parameters(), lr=lr) for i in range(n_nodes)]

# 目标网络同步
def update_targets():
    for i in range(n_nodes):
        targets[i].load_state_dict(agents[i].state_dict())

# ε-greedy 动作选择
def select_action(agent, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        with torch.no_grad():
            q_values = agent(state)
            return q_values.argmax().item()

# 训练单个智能体
def train_agent(agent, target, optimizer, batch):
    states, actions, rewards, next_states = zip(*batch)
    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.stack(next_states)
    
    q_values = agent(states)
    current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    with torch.no_grad():
        next_q_values = target(next_states)
        max_next_q = next_q_values.max(1)[0]
        expected_q = rewards + gamma * max_next_q

    loss = F.mse_loss(current_q, expected_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# 训练循环
update_targets()
for episode in range(1000):
    data = get_gat_data(opinions)
    with torch.no_grad():
        embeddings = gat_encoder(data)

    actions = []
    states = []
    for i in range(n_nodes):
        state = embeddings[i].detach()
        action = select_action(agents[i], state, epsilon)
        actions.append(action - 1)
        states.append(state)

    next_opinions = update_opinions(G, opinions, actions)
    
    # 奖励函数：共识奖励 + 个体奖励 + 邻居多样性奖励
    rewards = []
    for i in range(n_nodes):
        consensus_reward = -np.std(next_opinions)

        neighbors = list(G.neighbors(i))
        neighbor_sim = np.mean([next_opinions[j] for j in neighbors]) if neighbors else next_opinions[i]
        individual_reward = 1 - abs(next_opinions[i] - neighbor_sim)

        neighbor_opinions = np.array([next_opinions[j] for j in neighbors])
        neighbor_diversity = min(0.5, np.std(neighbor_opinions) * 2)

        rewards.append(0.4 * consensus_reward + 0.4 * individual_reward + 0.2 * neighbor_diversity)

    next_data = get_gat_data(next_opinions)
    with torch.no_grad():
        next_embeddings = gat_encoder(next_data)
    
    for i in range(n_nodes):
        memory.append((states[i], actions[i] + 1, rewards[i], next_embeddings[i]))

    if len(memory) > batch_size:
        batch = random.sample(memory, batch_size)
        for i in range(n_nodes):
            train_agent(agents[i], targets[i], optimizers[i], batch)

    opinions = next_opinions
    epsilon = max(0.1, epsilon * epsilon_decay)
    
    if episode % 10 == 0:
        update_targets()
        for node in range(n_nodes):
            print(f"Opinion of node {node}: {opinions[node]}")
        print(f"Episode {episode}, Epsilon: {epsilon:.2f}, Avg Opinion: {np.mean(opinions):.4f}, Std: {np.std(opinions):.4f}")