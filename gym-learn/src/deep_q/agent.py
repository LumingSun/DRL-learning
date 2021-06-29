import math
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
class ReplayBuffer():
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.idx =0
        self.buffer = []

    def push(self,state,action,reward,next_state,done):
        # if(len(self.buffer)<self.capacity):
        #     self.buffer.append((state,action,reward,next_state,done))
        # else:
        #     self.buffer[self.idx % self.capacity] = (state,action,reward,next_state,done)
        # self.idx += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.idx] = (state, action, reward, next_state, done)
        self.idx = (self.idx + 1) % self.capacity

    def batch(self,batch_size):
        batch = random.sample(self.buffer,batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

class MLP(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=512):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,output_dim)
    
    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
        

class DQN(object):
    def __init__(self,state_dim,action_dim,cfg):
        self.action_dim = action_dim
        # print("action dim: ", self.action_dim)
        self.state_dim = state_dim
        # print("state dim: ",self.state_dim)
        self.gamma = cfg.gamma
        self.device = cfg.device
        self.frame_idx = 0
        self.epsilon = lambda frame_idx: cfg.epsilon_end + (cfg.epsilon_end-cfg.epsilon_start) * math.exp(-1 * frame_idx/cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        
        self.policy_net = MLP(self.state_dim,self.action_dim,cfg.hidden_dim).to(self.device)
        self.target_net = MLP(self.state_dim,self.action_dim,cfg.hidden_dim).to(self.device)

        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        

    def choose_action(self, state):
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            action = self.predict(state)
        else:
            action = random.randrange(self.action_dim)
        return action

    def predict(self,state):
        with torch.no_grad():
            state = torch.tensor([state],device=self.device,dtype=torch.float32)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item()
        return action

    def update(self):
        if(len(self.memory)<self.batch_size):
            return
        state, action, reward, next_state, done = self.memory.batch(self.batch_size)

        state = torch.tensor(state,device=self.device,dtype=torch.float)
        action = torch.tensor(action,device=self.device).unsqueeze(1)
        next_state = torch.tensor(next_state,device=self.device,dtype=torch.float)
        reward = torch.tensor(reward,device=self.device,dtype=torch.float)
        done = torch.tensor(np.float32(done),device=self.device)
        q_values = self.policy_net(state).gather(dim=1, index=action)
        next_q_values = self.target_net(next_state).max(1)[0].detach()

        expected_q_values = reward + self.gamma * next_q_values * (1-done)
        
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path+'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path+'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)