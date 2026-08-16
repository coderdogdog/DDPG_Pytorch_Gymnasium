# DDPG 深度强化学习（PyTorch + Gymnasium）

基于 **PyTorch** 在 **Gymnasium** 环境下实现的 **DDPG（Deep Deterministic Policy Gradient，深度确定性策略梯度）** 算法，用于解决**连续动作空间**的深度强化学习问题。

## 项目简介

DDPG 是一种基于 Actor-Critic 框架的确定性策略梯度算法，结合了 DQN 的经验回放与目标网络思想，适用于动作空间连续的机器人控制、运动控制等任务。

本项目主要特点：

- 完整的 DDPG 实现：Actor（策略网络）与 Critic（价值网络）
- 目标网络软更新（soft update）
- 经验回放池（Replay Buffer）
- 高斯噪声探索 + 噪声衰减
- 梯度裁剪、LayerNorm / GELU 稳定训练
- 支持断点续训（checkpoint）
- TensorBoard 训练过程可视化
- 训练 / 测试两种运行模式

## 目录结构

```
DDPG_Pytorch_Gymnasium/
├── main.py        # 入口：解析命令行参数，区分训练 / 测试模式
├── agent.py       # ReplayBuffer（经验回放池）+ DDPGAgent（DDPG 智能体）
├── net.py         # Actor / Critic 网络结构定义
├── train.py       # 训练主循环：交互、更新、评估、保存、断点续训
├── test.py        # 测试：加载模型权重并评估 / 渲染
├── utils.py       # 工具函数：评估、随机种子固定、参数写入
├── model/         # 保存的模型权重文件
├── runs/          # TensorBoard 训练日志
└── README.md
```

## 环境依赖

- Python 3.11.15
- gymnasium 1.3.0
- torch 2.13.0（cu132 Nightly 版本）
- tensorboard（`torch.utils.tensorboard` 依赖）

> 显卡为 RTX 5060 时，PyTorch 建议安装最新的 Nightly 版本（cu132）：

```bash
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132
```

安装其余依赖：

```bash
pip install gymnasium tensorboard
```

## 快速开始

### 训练

```bash
python main.py --test_mode False --env "InvertedPendulum-v5"
```

也可以直接运行（`test_mode` 默认为 `False`）：

```bash
python main.py --env "Hopper-v5"
```

### 测试

```bash
python main.py --test_mode True --env "InvertedPendulum-v5" \
    --load_critic_name critic_train_10000.pth \
    --load_actor_name actor_train_10000.pth
```

## 命令行参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `cuda:0` | 训练设备 |
| `--train_seed` | `22` | 训练环境随机种子 |
| `--test_mode` | `False` | 是否开启测试模式 |
| `--test_seed` | `66` | 测试环境随机种子 |
| `--test_num` | `5` | 测试回合数 |
| `--test_render` | `human` | 测试渲染模式 |
| `--load_critic_name` | `critic_train_10000.pth` | 导入的 Critic 模型文件名 |
| `--load_actor_name` | `actor_train_10000.pth` | 导入的 Actor 模型文件名 |
| `--env` | `Hopper-v5` | 训练 / 测试环境 |
| `--a_bound` | `1.0` | 动作边界（例如 `[-1, 1]`） |
| `--warmup_steps` | `5000` | 预热步数（随机动作积累经验） |
| `--max_env_steps` | `1000000` | 环境最大运行总步数 |
| `--eval_interval` | `1000` | 评估间隔（更新次数） |
| `--save_interval` | `5000` | 模型保存间隔（更新次数） |
| `--hidden_dim` | `256` | 网络隐藏层大小 |
| `--buffer_max_len` | `1000000` | 经验回放池长度 |
| `--batch_size` | `256` | 训练 batch size |
| `--lr_actor` | `1e-4` | 策略网络学习率 |
| `--lr_critic` | `1e-3` | 价值网络学习率 |
| `--gamma` | `0.99` | 折扣因子 |
| `--noise_std` | `0.3` | 探索噪声标准差 |
| `--noise_decay` | `0.99995` | 噪声衰减系数（`=1` 表示不衰减） |
| `--tau` | `0.005` | 目标网络软更新系数 |
| `--clip_actor_norm` | `0.5` | 策略网络梯度裁剪阈值 |
| `--clip_critic_norm` | `1.0` | 价值网络梯度裁剪阈值 |

## 支持的环境

适用于以下 Gymnasium 连续动作空间环境（其他连续动作空间环境同样适用）：

- `Pendulum-v1`
- `InvertedPendulum-v5`
- `InvertedDoublePendulum-v5`
- `Reacher-v5`
- `Pusher-v5`
- `BipedalWalker-v3`
- `Hopper-v5`

## 核心实现说明

### Actor 网络

策略网络，根据状态输出确定性动作：

```
Linear -> LayerNorm -> GELU -> Linear -> LayerNorm -> GELU -> Linear -> Tanh
```

最终动作经 `Tanh` 归一化后乘以动作边界 `action_bound`，映射到环境允许的动作范围。

### Critic 网络

价值网络，采用状态塔 + 动作塔的双塔结构，融合后输出 Q 值：

```
state_net:  Linear -> LayerNorm -> GELU -> Linear -> LayerNorm -> GELU
action_net: Linear
fusion_net: Linear -> LayerNorm -> GELU -> Linear（输出 Q 值）
```

状态特征与动作特征拼接后融合，输出层使用极小权重初始化（`gain=0.01`），使初始 Q 值接近 0，稳定训练起步。

### DDPG 关键机制

1. **经验回放池**：预分配 NumPy 数组存储 `(state, action, reward, next_state, done)`，随机采样打破数据相关性。
2. **目标网络软更新**：`target = tau * online + (1 - tau) * target`，让目标网络缓慢逼近主网络。
3. **Critic 更新**：基于贝尔曼方程计算 TD 目标，使用 `smooth_l1_loss` 回归。
4. **Actor 更新**：最大化 `Q(state, actor(state))`，通过链式法则将梯度从 Critic 回传到 Actor。
5. **探索噪声**：高斯噪声叠加到动作上，随训练逐步衰减。
6. **梯度裁剪**：防止梯度爆炸。

## 日志与模型

### TensorBoard 日志

训练日志保存在 `runs/{env_name}/DDPG_{时间戳}/` 目录下，可用 TensorBoard 查看：

```bash
tensorboard --logdir runs
```

### 模型保存

- **定时保存**：每 `save_interval` 次更新保存 `actor_train_{N}.pth` 与 `critic_train_{N}.pth`。
- **最优模型**：评估奖励提升时保存 `best_actor_{avg_r}.pth` 与 `best_critic_{avg_r}.pth`。
- **断点续训**：保存 `checkpoint.pth`，包含网络、目标网络、优化器及训练状态；下次训练时若检测到该文件会自动从断点继续。

### 训练参数记录

每次训练的完整参数会写入对应日志目录下的 `training_parameters.txt`，方便复现实验。
