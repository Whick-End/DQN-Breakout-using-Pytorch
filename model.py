import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randint
from collections import namedtuple
import matplotlib.pyplot as plt
from main import args
import numpy as np
import random


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

class NeuralNetwork(nn.Module):
    
    def __init__(self, input_space, action_space):
        super(NeuralNetwork, self).__init__()
        """
        Neurons
        128 --> 256 --> 256 --> 3
        """
        self.fc1 = nn.Linear(input_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_space)

    
    def forward(self, x):
        y = F.leaky_relu(self.fc1(x))
        y = F.leaky_relu(self.fc2(y))

        # Output layer
        return self.fc3(y)
           
class ReplayMemory(object):
    """
    Store history,
    To learn about it later
    """
    
    def __init__(self, capacity, batch_size):
        self.memory = []
        self.capacity = capacity
        self.min_learn = capacity / 10
        self.batch_size = batch_size
        self.transition = namedtuple('Memory', ('state', 'action', 'reward', 'next_state'))
        
    def push_memory(self, frames, action, reward, frames_):
        if len(self.memory) <= self.capacity:
            self.memory.append(self.transition(frames, action, reward, frames_))
            
        else:
            del self.memory[0]
            self.memory.append(self.transition(frames, action, reward, frames_))

    def memory_rand(self):
        """
        Get random batch_size from replay memory
        """
        state, action, reward, next_state = zip(*random.sample(self.memory, self.batch_size))
        state = torch.stack(state).to(device)
        next_state = torch.stack(next_state).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        return state, action, reward, next_state
    
    def clear(self):
        self.memory = []
    
    def __len__(self):
        return len(self.memory)
    
def convert_np(observation):
    if device.type == 'cuda':
            return torch.cuda.FloatTensor(observation)/255
    return torch.FloatTensor(observation)/255

    
class Agent(nn.Module):
    
    def __init__(self, memory, input_space, action_space):
        super(Agent, self).__init__()
        self.memory = memory
        
        self.train_model = NeuralNetwork(input_space, action_space)
        self.train_model.apply(self.normalize_weight)

        self.target_model = NeuralNetwork(input_space, action_space)
        self.target_model.load_state_dict(self.train_model.state_dict())
        
        if device.type == 'cuda':
            self.train_model.cuda()
            self.target_model.cuda()
        
        self.optimizer = torch.optim.RMSprop(params=self.train_model.parameters(),
                                                lr=args.learning_rate)
 
        self.input_space = input_space
        self.action_space = action_space
        
        self.epsilon = args.epsilon
        self.start_epsilon = self.epsilon
        self.min_epsilon = args.min_epsilon
        self.episode_step = 5000
    
    def save(self):
        """
        Save model
        """
        torch.save({"state_dict": self.train_model.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                    }, args.saved_as)
    
    def load(self, name):
        """
        Load model
        """
        checkpoint = torch.load(name)
        self.train_model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    
    def take_action(self, frame):
        """
        Use epsilon greedy strategy,
        To take action
        """
        random_number = torch.rand(1)
        
        if random_number <= self.epsilon:
            action = randint(0, self.action_space-1)
            
        else:
            if device.type == 'cuda':
                frame = torch.cuda.FloatTensor(frame)
            else:
                frame = torch.FloatTensor(frame)
            action = self.train_model(frame.unsqueeze(0))
            action = action.max(1)[1].item()
        
        return action
 
    def learn(self):
        """
        Learning using Deeq Q Learning
        target = reward + GAMMA * q_next
        """
        frames, actions, rewards, frames_ = self.memory.memory_rand()

        # Get value of actions
        q_values = self.train_model(frames)
        q_values = torch.gather(q_values, index=actions.unsqueeze(1), dim=1)
        
        # Get q_next values
        q_next = self.target_model(frames_).detach()
        q_next = q_next.max(1)[0]
        
        # Bellman equation
        target = rewards + args.gamma * q_next
        target = target.view(32, 1)
        
        # Get loss
        loss = (target - q_values).pow(2).sum().div(2)
        
        self.optimizer.zero_grad()
        # Gradient
        loss.backward()
        
        for p in self.train_model.parameters():
            p.grad.data.clamp_(-1.0, 1.0)

        self.optimizer.step()
        
    def normalize_weight(self, m):        
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            
    def should_learn(self):
        """
        Return if the learn function, can be call
        """
        if len(self.memory) > self.memory.min_learn:
            return True
        
        return False
            
    def update(self, episode):
        """
        Update target model, and epsilon
        """
        if episode % 50 == 0:
            self.target_model.load_state_dict(self.train_model.state_dict())
        
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.min_epsilon + (self.start_epsilon - self.min_epsilon) * torch.exp(
                torch.tensor(-1. * episode / self.episode_step))

    def eval(self):
        """
        Set eval mode
        """
        self.train_model.eval()
        self.epsilon = 0.0
        
    def plot(self, history):
        """
        Show graph of performance
        Really useful in learning
        """
        if len(history) > 10:
            del history[0]

        x, y = zip(*history)
        plt.plot(x, y, color='r')
        plt.title('Score per episode')
        plt.xlabel('Episodes')
        plt.ylabel('Scores')
        plt.draw()
        plt.pause(1e-8)
