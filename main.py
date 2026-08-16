# -*- coding: utf-8 -*-

"""
基于 Pytorch 在 Gymnasium 环境下
深度强化学习算法 DDPG 处理连续动作空间问题

python 3.11.15
gymnasium 1.3.0
torch 2.13.0.dev20260611+cu132

显卡：5060 pytorch 应该安装最新Nightly 版本
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132

训练：
python main.py --test_mode False --env "InvertedPendulum-v5"

测试：
python main.py --test_mode True --env "InvertedPendulum-v5"
--load_critic_name critic_train_10000.pth
--load_actor_name actor_train_10000.pth

"""


import argparse
from test import test_agent
from train import train_agent


def parse_args():
    parser = argparse.ArgumentParser(description="DDPG On Gymnasium")

    parser.add_argument("--device", type=str, default="cuda:0", help="训练设备: --device cuda:0")
    parser.add_argument("--train_seed", type=int, default=22, help="训练环境种子: --train_seed 22")
    parser.add_argument("--test_mode", type=bool, default=False, help="是否设置为测试模式: --test_mode False")

    # 测试
    parser.add_argument("--test_seed", type=int, default=66, help="测试环境种子: --test_seed 66")
    parser.add_argument("--test_num", type=int, default=5, help="测试环境次数: --test_num 5")
    parser.add_argument("--test_render", type=str, default="human", help="测试环境渲染模式: --test_render human")
    parser.add_argument("--load_critic_name", type=str, default="critic_train_10000.pth", help="导入critic模型文件")
    parser.add_argument("--load_actor_name", type=str, default="actor_train_10000.pth", help="导入actor模型文件")

    # 环境
    parser.add_argument("--env", type=str, default="Hopper-v5",
                        help="Pendulum-v1, InvertedPendulum-v5, InvertedDoublePendulum-v5")
    parser.add_argument("--a_bound", type=float, default=1.0, help="动作边界 例如:[-1, 1]")

    # 训练
    parser.add_argument("--warmup_steps", type=int, default=5000, help="先预热 随机选取动作积累经验 先走几步")
    parser.add_argument("--max_env_steps", type=int, default=1000000, help="环境最大运行总步数")
    parser.add_argument("--eval_interval", type=int, default=1000, help="评估间隔（智能体更新 eval_steps 次后评估模型）")
    parser.add_argument("--save_interval", type=int, default=5000, help="保存模型间隔（智能体更新 save_steps 步后保存模型）")

    # 超参数
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="网络的隐藏层大小，例如：--hidden_dim 256")

    parser.add_argument("--buffer_max_len", type=int, default=int(1e6), help="经验回放池长度")
    parser.add_argument("--batch_size", type=int, default=256, help="训练时batch_size")
    parser.add_argument("--lr_actor", type=float, default=1e-4, help="策略网络学习率")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="Q网络学习率")

    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--noise_std", type=float, default=0.3, help="noise_std 帮助探索,防止过早掉入局部最优")
    parser.add_argument("--noise_decay", type=float, default=0.99995, help="噪音衰减 或者=1不衰减")
    parser.add_argument("--tau", type=float, default=0.005, help="滑动更新")

    parser.add_argument("--clip_actor_norm", type=float, default=0.5, help="策略网络梯度裁剪阈值")
    parser.add_argument("--clip_critic_norm", type=float, default=1.0, help="价值网络梯度裁剪阈值")

    return parser.parse_args()


def main():

    args = parse_args()

    if args.test_mode:
        avg_scores, avg_steps = test_agent(args)
        print(f"回合平均奖励: {avg_scores} | 回合平均步数: {avg_steps}")

    else:
        train_agent(args)

if __name__ == "__main__":
    main()
