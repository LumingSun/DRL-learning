import torch
import numpy as np
import math

class QLearning(object):
    """Q-learning algorithm with e-greedy policy
    """
    def __init__(self,state_dim,action_dim,cfg):
        self.Q_table = np.zeros((state_dim,action_dim))
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = cfg.lr
        self.gamma = cfg.gamma
        self.epsilon = 0
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.sample_count = 0


    def choose_action(self,state):
        """return an action according to the state

        Args:
            state ([type]): [description]
        """
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1 * self.sample_count / self.epsilon_decay)
        
        if np.random.uniform(0,1) > self.epsilon:
            action = self.predict(state)
        else:
            action = np.random.choice(self.action_dim)

        return action


    def predict(self,state):
        """return the action with max value of current state

        Args:
            state ([type]): [description]
        """
        action = np.argmax(self.Q_table[state])
        return action

    def update(self,state,action,next_state,reward,done):
        Q_predict = self.Q_table[state][action]
        if done:
            Q_target = reward   
        else:
            Q_target = reward + self.gamma * np.max((self.Q_table[next_state]))
        self.Q_table[state][action] += self.lr * (Q_target - Q_predict)
    
    def save(self,path):
        np.save(path+"Q_table.npy",self.Q_table)

    def load(self,path):
        self.Q_table = np.load(path+"Q_table.npy")

