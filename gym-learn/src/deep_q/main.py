import gym
import torch
from agent import DQN
from utils import plot_losses,plot_rewards

class DQNConfig():
    def __init__(self):
        self.env = 'CartPole-v0'
        self.algo = 'DQN'

        self.lr = 0.0001
        self.train_eps = 300
        self.eval_eps = 50
        self.gamma = 0.95
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = './dqn_model'
        self.result_path = './dqn_result'

        self.epsilon_start = 0.90  # start epsilon of e-greedy policy
        self.epsilon_end = 0.01
        self.epsilon_decay = 500

        self.batch_size = 64
        self.memory_capacity = 100000
        self.target_update = 2 # update frequency of target net
        self.hidden_dim = 256  # hidden size of net

def env_agent_config(cfg:DQNConfig,seed=0):
    env = gym.make(cfg.env)
    env.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim,action_dim,cfg)

    return env,agent

def train(cfg:DQNConfig,env,agent:DQN):
    rewards = []
    ma_rewards = []
    for eps in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while True:
            action = agent.choose_action(state)
            # print(state)
            next_state, reward, done, _ = env.step(action)
            # print(next_state)
            ep_reward += reward
            agent.memory.push(state,action,reward,next_state,done)
            state = next_state
            agent.update()
            if done:
                break
        if (eps+1) % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        if (eps+1)%10 == 0:
            print('Episode:{}/{}, Reward:{}'.format(eps+1, cfg.train_eps, ep_reward))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    return rewards, ma_rewards

def eval(cfg:DQNConfig,env,agent:DQN):
    rewards = []
    ma_rewards = []
    for eps in range(cfg.eval_eps):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9 + ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        print("Episode:{i_ep+1}/{cfg.eval_eps}, reward:{ep_reward:.1f}")
    print('Complete evaling！')
    return rewards,ma_rewards



if __name__ == "__main__":
    cfg = DQNConfig()
    env, agent = env_agent_config(cfg)
    rewards, ma_rewards = train(cfg, env, agent)
    agent.save(path=cfg.model_path)

    plot_rewards(rewards,ma_rewards,tag="train",env=cfg.env,algo = cfg.algo,path=cfg.result_path)

    env,agent = env_agent_config(cfg)
    agent.load(path=cfg.model_path)
    rewards,ma_rewards = eval(cfg,env,agent)
    plot_rewards(rewards,ma_rewards,tag="eval",env=cfg.env,algo = cfg.algo,path=cfg.result_path)
