# -*- coding: utf-8 -*-
import gymnasium as gym
from pathlib import Path
from agent import DDPGAgent
from utils import evaluate_agent


def test_agent(args):
    # 创建环境
    env_name = args.env
    test_env = gym.make(env_name, render_mode=args.test_render)

    # 测试环境随机种子
    test_env.reset(seed=args.test_seed)

    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.shape[0]
    action_bound = args.a_bound

    # 创建智能体
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

    model_critic_path = f"./model/{env_name}/" + args.load_critic_name
    model_actor_path = f"./model/{env_name}/" + args.load_actor_name
    Path(model_critic_path).parent.mkdir(parents=True, exist_ok=True)
    Path(model_actor_path).parent.mkdir(parents=True, exist_ok=True)

    ddpg_agent.load(model_critic_path, model_actor_path)

    is_render_human = False
    if args.test_render == "human":
        is_render_human = True

    avg_scores, avg_steps = evaluate_agent(test_env, ddpg_agent, is_render_human, args.test_num)
    test_env.close()

    return avg_scores, avg_steps
