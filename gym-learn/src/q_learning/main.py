import gym
import torch
from agent import QLearning
from utils import plot_losses,plot_rewards


class QlearningConfig:
    def __init__(self):
        self.algo = "Q-Learning"
        self.env = "CliffWalking-v0"
        self.lr = 0.1

        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 200

        self.gamma = 0.9
        
        self.train_eps = 300
        self.eval_eps = 30
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = "./q_learning_model"
        self.result_path = "./q_learning_result"

def env_agent_config(cfg:QlearningConfig):
    env = gym.make(cfg.env)

    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    
    agent = QLearning(state_dim,action_dim,cfg)
    return env,agent

def train(cfg:QlearningConfig,env,agent:QLearning):
    rewards = []
    moving_average_rewards = []
    
    for i_ep in range(cfg.train_eps):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.update(state,action,next_state,reward,done)
            state = next_state
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if moving_average_rewards:
            moving_average_rewards.append(moving_average_rewards[-1]*0.9 + ep_reward*0.1)
        else:
            moving_average_rewards.append(ep_reward)
        print("Episode:{}/{}: reward:{:.1f}".format(i_ep+1, cfg.train_eps,ep_reward))
    print('Complete training！')
    return rewards,moving_average_rewards

def eval(cfg:QlearningConfig,env,agent:QLearning):
    rewards = []
    moving_average_rewards = []
    
    for i_ep in range(cfg.eval_eps):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state,reward,done,_ = env.step(action)
            state = next_state
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if moving_average_rewards:
            moving_average_rewards.append(moving_average_rewards[-1]*0.9 + ep_reward*0.1)
        else:
            moving_average_rewards.append(ep_reward)
        print("Episode:{i_ep+1}/{cfg.eval_eps}, reward:{ep_reward:.1f}")
    print('Complete evaling！')
    return rewards,moving_average_rewards

if __name__ == "__main__":
    cfg = QlearningConfig()
    env,agent = env_agent_config(cfg)
    rewards,ma_rewards = train(cfg,env,agent)
    agent.save(path=cfg.model_path)
    plot_rewards(rewards,ma_rewards,tag="train",env=cfg.env,algo = cfg.algo,path=cfg.result_path)

    env,agent = env_agent_config(cfg)
    agent.load(path=cfg.model_path)
    rewards,ma_rewards = eval(cfg,env,agent)
    plot_rewards(rewards,ma_rewards,tag="eval",env=cfg.env,algo = cfg.algo,path=cfg.result_path)
