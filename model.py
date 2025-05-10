import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# 图注意力网络（GAT）聚合邻居信息
class GATEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8):
        super(GATEncoder, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4)  # 多头注意力
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim)      # 合并多头输出
        self.out_dim = hidden_dim  # 保存输出维度，便于外部访问

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x  # 输出节点嵌入

# 单智能体 DQN 网络（输入：状态嵌入 → 输出：Q 值）
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
        # 确保输入维度正确
        if x.dim() == 1:
            x = x.unsqueeze(0)  # 单样本添加批次维度
        
        # 检查输入维度并调整
        if x.size(-1) != self.state_dim:
            # 使用线性映射将输入调整为正确的维度
            x = F.pad(x, (0, self.state_dim - x.size(-1))) if x.size(-1) < self.state_dim else x[:, :self.state_dim]
            
        return self.net(x)