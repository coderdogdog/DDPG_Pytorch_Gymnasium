# -*- coding: utf-8 -*-
import gymnasium as gym
from pathlib import Path
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
from agent import DDPGAgent, ReplayBuffer
from utils import evaluate_agent, args_to_txt, set_seed


def train_agent(args):
    # 创建模型文件夹
    # ./model/env_name/
    model_dir = f"./model/{args.env}/"
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # 训练日志
    log_dir = f'''runs/{args.env}/DDPG_{datetime.now().strftime("%Y%m%d_%H_%M_%S")}'''
    writer = SummaryWriter(log_dir)

    # 当前训练参数写入 txt 文件，方便查看
    args_txt_path = log_dir + "/training_parameters.txt"
    args_to_txt(args, args_txt_path)


    print("=" * 100)
    print("------------- 深度强化学习算法 DDPG 处理连续动作空间问题 -------------")
    print('''Gymnasium 环境（连续动作空间）:
        "Pendulum-v1",
        "InvertedPendulum-v5",
        "InvertedDoublePendulum-v5",
        "Reacher-v5",
        "Pusher-v5",
        "BipedalWalker-v3"
        [其他连续动作空间的游戏环境也可以]
        ''')
    print(f'本次实验环境: {args.env}')
    print(f'本次实验的训练参数在: {log_dir} 下的 training_parameters.txt 中查看.')
    print("=" * 100)

    print(f"[环境] {args.env} ")
    print("\n")

    # *********************************************************************************************************
    # *********************************************************************************************************
    # *********************************************************************************************************

    # 创建训练环境
    env = gym.make(args.env, render_mode=None)
    # 创建用来测试评估的环境
    evaluate_env = gym.make(args.env, render_mode=None)

    # 设置随机种子
    set_seed(args.train_seed)
    env.reset(seed=args.train_seed)
    evaluate_env.reset(seed=args.test_seed)

    # 游戏环境信息
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = args.a_bound

    buffer = ReplayBuffer(args.buffer_max_len, state_dim, action_dim)

    ddpg_agent = DDPGAgent(state_dim, action_dim, args.hidden_dim, action_bound,
                           gamma=args.gamma,
                           lr_critic=args.lr_critic,
                           lr_actor=args.lr_actor,
                           noise_std=args.noise_std,
                           noise_decay=args.noise_decay,
                           tau=args.tau,
                           clip_critic_norm=args.clip_critic_norm,
                           clip_actor_norm=args.clip_actor_norm,
                           device=args.device)

    # ---------------------------------------------- 训练状态 ------------------------------------------------------

    total_steps = 0
    # ------------------------------------- 断点续训 ------------------------------------
    checkpoint_path = model_dir + "checkpoint.pth"
    resume = os.path.isfile(checkpoint_path)

    if resume:
        ddpg_agent.train_num, total_steps, ddpg_agent.noise_std, best_avg_reward \
            = ddpg_agent.load_checkpoint(checkpoint_path)
        # 跳转到对应步数
        print(f"[续训] 从智能体第 {ddpg_agent.train_num} 次更新继续训练")

    else:
        best_avg_reward = -float(10000)
        print("[新训练] 从头开始训练")

    best_critic_path = model_dir + "best_critic_" + str(int(best_avg_reward)) + ".path"
    best_actor_path = model_dir + "best_actor_" + str(int(best_avg_reward)) + ".path"

    # ------------------------------------- 智能体开始与环境交互 ------------------------------------
    # 游戏回合前初始化
    state, _ = env.reset()

    try:
        while total_steps < args.max_env_steps:
            # 先预热 随机选取动作积累经验 先走 10000 步
            if total_steps < args.warmup_steps:
                action = env.action_space.sample()  # 预热期随机探索
            else:
                action = ddpg_agent.select_action(state)

            next_state, reward, terminated, truncated, infos = env.step(action)
            done = terminated or truncated
            # 特殊环境的奖励缩放
            if args.env == "Pendulum-v1":
                reward = float(reward) / 10

            buffer.store(state, action, reward, next_state, terminated)
            total_steps += 1

            if done:
                state, _ = env.reset()
            else:
                state = next_state

            if total_steps > args.warmup_steps:
                ddpg_agent.update(buffer, args.batch_size, writer)
                # ----------------------------------- 测试 ---------------------------------------------------
                if ddpg_agent.train_num % args.eval_interval == 0:
                    avg_r, avg_steps = evaluate_agent(evaluate_env, ddpg_agent,
                                                      is_render_human=False, test_numb=3)

                    print(f"游戏总步数: {total_steps:8d} | 网络更新次数: {ddpg_agent.train_num:8d} | "
                          f"每回合平均奖励: {avg_r:9.2f} | 每回合平均步数: {avg_steps:5d}")

                    writer.add_scalar("test/avg_r", avg_r, ddpg_agent.train_num)
                    writer.add_scalar("test/avg_steps", avg_steps, ddpg_agent.train_num)

                    if avg_r > best_avg_reward:
                        if os.path.isfile(best_critic_path):
                            os.remove(best_critic_path)

                        if os.path.isfile(best_actor_path):
                            os.remove(best_actor_path)

                        best_avg_reward = avg_r

                        best_critic_path = model_dir + "best_critic_" + str(int(best_avg_reward)) + ".pth"
                        best_actor_path = model_dir + "best_actor_" + str(int(best_avg_reward)) + ".pth"

                        ddpg_agent.save(best_critic_path, best_actor_path)

                # # ----------------------------------- 定时保存模型 ---------------------------------------------------
                if ddpg_agent.train_num % args.save_interval == 0:
                    # critic_train_10000.pth
                    critic_name = "critic_train_" + str(ddpg_agent.train_num) + ".pth"
                    actor_name = "actor_train_" + str(ddpg_agent.train_num) + ".pth"

                    critic_path = model_dir + critic_name
                    actor_path = model_dir + actor_name
                    ddpg_agent.save(critic_path, actor_path)
                    # 覆盖式检查点（用于断点续训）
                    ddpg_agent.save_checkpoint(checkpoint_path, ddpg_agent.train_num, total_steps,
                                               ddpg_agent.noise_std, best_avg_reward)

    except KeyboardInterrupt:
        print("\n\n[中断] 用户手动停止训练")

    env.close()
    evaluate_env.close()
    writer.close()

