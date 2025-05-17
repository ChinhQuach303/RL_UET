import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt


class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_min=0.01, 
                 epsilon_decay=0.998, learning_rate=1e-4, batch_size=128, memory_capacity=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Separate networks for movement and package actions
        self.movement_net = self._build_movement_network().to(self.device)
        self.package_net = self._build_package_network().to(self.device)
        
        self.target_movement_net = self._build_movement_network().to(self.device)
        self.target_package_net = self._build_package_network().to(self.device)
        
        self.target_movement_net.load_state_dict(self.movement_net.state_dict())
        self.target_package_net.load_state_dict(self.package_net.state_dict())
        
        self.target_movement_net.eval()
        self.target_package_net.eval()

        self.optimizer = optim.Adam([
            {'params': self.movement_net.parameters()},
            {'params': self.package_net.parameters()}
        ], lr=learning_rate)

    def _build_movement_network(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 5)  # 5 movement actions
        )

    def _build_package_network(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # 3 package actions
        )

    def select_action(self, state, bfs_action=None, bfs_prob=0.2):
        if bfs_action is not None and random.random() < bfs_prob:
            return bfs_action
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if random.random() > self.epsilon:
            with torch.no_grad():
                movement_q = self.movement_net(state)
                package_q = self.package_net(state)
                movement_action = movement_q.argmax().item()
                package_action = package_q.argmax().item()
                return movement_action, package_action
        else:
            return random.randrange(5), random.randrange(3)

    def store_transition(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
            
        transitions = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*transitions)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # Split actions into movement and package actions
        movement_actions = torch.LongTensor([a[0] for a in action_batch]).to(self.device)
        package_actions = torch.LongTensor([a[1] for a in action_batch]).to(self.device)

        # Get current Q values
        movement_q_values = self.movement_net(state_batch).gather(1, movement_actions.unsqueeze(1)).squeeze(1)
        package_q_values = self.package_net(state_batch).gather(1, package_actions.unsqueeze(1)).squeeze(1)

        # Get next Q values
        with torch.no_grad():
            next_movement_q = self.target_movement_net(next_state_batch).max(1)[0]
            next_package_q = self.target_package_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.gamma * (next_movement_q + next_package_q) * (1 - done_batch)

        # Compute loss
        movement_loss = nn.MSELoss()(movement_q_values, target_q_values)
        package_loss = nn.MSELoss()(package_q_values, target_q_values)
        loss = movement_loss + package_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.movement_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.package_net.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_movement_net.load_state_dict(self.movement_net.state_dict())
        self.target_package_net.load_state_dict(self.package_net.state_dict())