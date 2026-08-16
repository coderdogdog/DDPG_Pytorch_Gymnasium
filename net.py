# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn


def layer_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    """正交初始化线性层权重，偏置置零。"""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # ========== 1. 状态塔 (高维特征提取) ==========
        # 结构: Linear -> Norm -> ReLU -> Linear -> Norm -> ReLU
        # 核心修改：在激活函数之前加入归一化，能有效防止内部协变量偏移
        self.state_net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),  # 对隐藏层特征做归一化
            nn.GELU(),

            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),  # 第二层同样加归一化
            nn.GELU()
        )

        # ========== 2. 动作塔 (将低维动作映射到隐藏空间) ==========
        # 动作通常已经被 tanh 限制在 [-1, 1] 区间，且维度低，
        # 这里可以不额外加归一化（简单线性映射即可）
        # 为了保证输出到 hidden_dim 的特征方差稳定，建议使用gain = 1.0（因为不需要像 ReLU 那样放大方差）
        self.action_net = layer_init(nn.Linear(action_dim, hidden_dim), gain=1.0)

        # ========== 3. 融合层 ==========
        # 状态特征(256) + 动作特征(256) = 512，再次经过归一化后输出Q值
        self.fusion_net = nn.Sequential(
            layer_init(nn.Linear(hidden_dim * 2, hidden_dim)),
            nn.LayerNorm(hidden_dim),  # 融合后的特征也做归一化
            nn.GELU(),
            # 最后一层（输出层）：使用极小权重，初始 Q 值接近 0，稳定训练起步
            layer_init(nn.Linear(hidden_dim, 1), gain=0.01)  # 输出Q值，不加激活函数
        )

    def forward(self, state, action):
        state_feat = self.state_net(state)  # [batch, hidden_dim]
        action_feat = self.action_net(action)  # [batch, hidden_dim]
        combined = torch.cat([state_feat, action_feat], dim=-1)  # [batch, hidden_dim*2]
        q_value = self.fusion_net(combined)
        return q_value


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_bound=1.0):
        super(Actor, self).__init__()

        self.layers = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            # 如果和 critic 双重缩小 gain=0.01 可能会导致梯度消失
            layer_init(nn.Linear(hidden_dim, action_dim), gain=0.5),
            nn.Tanh()
        )
        self.action_bound = action_bound

    def forward(self, state):
        a = self.layers(state)
        action = a * self.action_bound
        return action

