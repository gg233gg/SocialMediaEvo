import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# 图注意力网络
class GATEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8):
        super(GATEncoder, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim)
        self.out_dim = hidden_dim  # 保存输出维度，便于外部访问

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

# DQN网络
class DQNAgent(nn.Module):
    def __init__(self, state_dim=8, action_dim=3):
        super(DQNAgent, self).__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # 检查输入维度并调整
        if x.size(-1) != self.state_dim:
            # 使用线性映射将输入调整为正确的维度
            x = F.pad(x, (0, self.state_dim - x.size(-1))) if x.size(-1) < self.state_dim else x[:, :self.state_dim]
            
        return self.net(x)