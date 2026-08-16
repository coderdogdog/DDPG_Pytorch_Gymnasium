# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from net import Critic, Actor


class ReplayBuffer:
    def __init__(self, max_len: int, state_dim: int, action_dim: int):
        self.max_len = max_len

        self.next_idx = 0           # 下一个要写入的位置
        self.count = 0              # 当前已有数据量

        # 为每个数据字段预分配 NumPy 数组
        self.states = np.zeros((max_len, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_len, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_len, 1), dtype=np.float32)
        self.next_states = np.zeros((max_len, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_len, 1), dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        idx = self.next_idx
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        self.count = min(self.count + 1, self.max_len)
        self.next_idx = (self.next_idx + 1) % self.max_len

    def sample(self, batch_size):
        """随机采样一个 batch"""
        # 从 [0, self.count) 范围内随机选取 batch_size 个索引
        # replace=False 的意思是：不放回抽样 不抽一样的
        indices = np.random.choice(self.count, batch_size, replace=False)
        # 直接从数组中根据索引取值
        return (self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_states[indices],
                self.dones[indices])


class DDPGAgent:
    def __init__(self, state_dim, action_dim, hid_dim, a_bound,
                 gamma=0.99,
                 lr_critic=1e-3,
                 lr_actor=3e-4,
                 noise_std=0.3,
                 noise_decay=0.9995,
                 tau=0.005,
                 clip_critic_norm=0.5,
                 clip_actor_norm=0.5,
                 device="cpu"):

        a_bound = torch.tensor(a_bound, dtype=torch.float32, requires_grad=False).to(device)

        # 初始化网络
        self.actor_net = Actor(state_dim, action_dim, hid_dim, a_bound).to(device)
        self.critic_net = Critic(state_dim, action_dim, hid_dim).to(device)

        # 初始化目标网络 (结构与主网络完全相同，初始参数复制)
        self.target_actor_net = Actor(state_dim, action_dim, hid_dim, a_bound).to(device)
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())

        self.target_critic_net = Critic(state_dim, action_dim, hid_dim).to(device)
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=lr_critic)

        self.action_dim = action_dim
        self.action_bound = a_bound

        # 参数
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # 目标网络软更新系数

        # 探索噪声参数
        # 如果 action_bound = 10.0，std = 0.1  就太小了（仅1 %），需要调大至 std = 1.0。
        self.noise_std = noise_std  # 高斯噪声的标准差（训练初期可以设大些，如0.2）
        self.noise_decay = noise_decay
        # 梯度裁剪
        self.clip_critic_norm = clip_critic_norm
        self.clip_actor_norm = clip_actor_norm

        self.train_num = 0
        self.dvc = device


    def select_action(self, state, explore=True):
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float32).to(self.dvc)
            action = self.actor_net(state)
            action = action.detach().cpu().numpy().flatten()

            if explore:
                noise = np.random.normal(0, self.noise_std, size=self.action_dim)
                action = action + noise
                # 裁剪动作，确保不超出环境允许的范围（物理限制）
                action = np.clip(action, -self.action_bound.item(), self.action_bound.item())

            return action

    def update(self, replay_buffer, batch_size, writer):

        # ---------- Step 1: 从经验池中采样 ----------
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        # 转换为 PyTorch Tensor
        state = torch.tensor(state, dtype=torch.float32).to(self.dvc)
        action = torch.tensor(action, dtype=torch.float32).to(self.dvc)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.dvc)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.dvc)
        done = torch.tensor(done, dtype=torch.float32).to(self.dvc)

        # ---------- Step 2: 更新 Critic (回归任务) ----------
        # Double DQN 的思想: target Qnet 只负责价值评估不参与动作选择
        # Actor 负责选择动作
        with torch.no_grad():  # 计算 Target Q 值时，不追踪梯度
            # 目标 Actor 根据下一状态选出动作
            next_action = self.target_actor_net(next_state)
            # 目标 Critic 计算下一状态的 Q 值
            q_next_sa = self.target_critic_net(next_state, next_action)
            # 贝尔曼方程计算 TD 目标 (标签)
            # 注意：如果 done==1 (终止状态)，则 Q_next 应为 0
            q_target = reward + (1 - done) * self.gamma * q_next_sa

        # 当前 Critic 对 (state, action) 的预测值
        q_current = self.critic_net(state, action)

        # 计算 Critic 损失 (MSE)
        # critic_loss = F.mse_loss(q_current, q_target)
        critic_loss = F.smooth_l1_loss(q_current, q_target, reduction='mean')

        # 反向传播更新 Critic 主网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 梯度裁剪，防止梯度爆炸
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=self.clip_critic_norm)
        self.critic_optimizer.step()

        # ---------- Step 3: 更新 Actor (策略梯度，最大化 Q 值) ----------
        # 冻结 critic，梯度只流向 actor
        for p in self.critic_net.parameters():
            p.requires_grad = False

        # 根据当前状态，让当前 Actor 生成新动作
        new_action = self.actor_net(state)
        # 计算 Actor 的损失：取负的 Q 值，做梯度下降来最大化 Q
        # 核心链式法则：PyTorch 自动微分会通过 Critic 网络回传到 Actor 网络
        actor_loss = -self.critic_net(state, new_action).mean()

        # 反向传播更新 Actor 主网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # 梯度裁剪，防止梯度爆炸
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=self.clip_actor_norm)
        self.actor_optimizer.step()

        # 恢复 critic 参数
        for p in self.critic_net.parameters():
            p.requires_grad = True

        # ---------- Step 4: 软更新目标网络 (Soft Update) ----------
        # 让目标网络的参数缓慢靠近主网络
        for param, target_param in zip(self.actor_net.parameters(), self.target_actor_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_net.parameters(), self.target_critic_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # 控制噪声衰减的快慢
        if self.train_num % 5 == 0:
            if self.noise_std > 0.02:
                self.noise_std *= self.noise_decay
            else:
                self.noise_std = 0.02

        self.train_num += 1

        if self.train_num % 10 == 0:
            # TensorBoard 记录
            writer.add_scalar("train/q_value", q_current.mean().item(), self.train_num)
            writer.add_scalar("train/critic_loss", critic_loss.item(), self.train_num)
            writer.add_scalar("train/actor_loss", actor_loss.item(), self.train_num)
            writer.add_scalar("train/noise_std", self.noise_std, self.train_num)

            writer.add_scalar("grad_norm/critic_grad_norm", critic_grad_norm, self.train_num)
            writer.add_scalar("grad_norm/actor_grad_norm", actor_grad_norm, self.train_num)


    def save(self, path_critic, path_actor):
        """保存模型权重"""
        torch.save(self.critic_net.state_dict(), path_critic)
        torch.save(self.actor_net.state_dict(), path_actor)

    def load(self, path_critic, path_actor):
        """加载模型权重"""
        self.critic_net.load_state_dict(torch.load(path_critic, map_location=self.dvc, weights_only=True))
        self.actor_net.load_state_dict(torch.load(path_actor, map_location=self.dvc, weights_only=True))

    # 断点续训保存
    def save_checkpoint(self, path, train_num, total_steps, noise_std, best_reward):
        torch.save({
            'model_critic_dict': self.critic_net.state_dict(),
            'model_actor_dict': self.actor_net.state_dict(),
            'target_critic_dict': self.target_critic_net.state_dict(),
            'target_actor_dict': self.target_actor_net.state_dict(),

            'optimizer_critic_dict': self.critic_optimizer.state_dict(),
            'optimizer_actor_dict': self.actor_optimizer.state_dict(),

            'train_num': train_num,
            'total_steps': total_steps,
            'noise_std': noise_std,
            'best_avg_reward': best_reward,
        }, path)

    # 断点续训加载
    def load_checkpoint(self, path):
        # 加载全部数据
        ckpt = torch.load(path, map_location=self.dvc, weights_only=True)

        # 加载主网络
        self.critic_net.load_state_dict(ckpt['model_critic_dict'])
        self.actor_net.load_state_dict(ckpt['model_actor_dict'])

        self.target_critic_net.load_state_dict(ckpt['target_critic_dict'])
        self.target_actor_net.load_state_dict(ckpt['target_actor_dict'])

        # 加载优化器
        self.critic_optimizer.load_state_dict(ckpt['optimizer_critic_dict'])
        self.actor_optimizer.load_state_dict(ckpt['optimizer_actor_dict'])

        # ===== 加载后清空残留梯度 =====
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        return ckpt['train_num'], ckpt['total_steps'], ckpt['noise_std'], ckpt['best_avg_reward']

