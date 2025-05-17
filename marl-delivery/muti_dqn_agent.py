import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from delivery_env import DeliveryEnv

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=1e-3, batch_size=32, memory_capacity=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy and target networks
        self.policy_net = self._build_model().to(self.device)
        self.target_net = self._build_model().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),  # Use state_size (86) instead of hard-coded 128
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = q_values.argmax().item()
        else:
            action = random.randrange(self.action_size)
        return action

    def store_transition(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*transitions)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

class MultiAgentDQN:
    def __init__(self, n_agents, state_size, gamma=0.99, epsilon=1.0, epsilon_min=0.01, 
                 epsilon_decay=0.998, learning_rate=1e-4, batch_size=128, memory_capacity=10000):
        self.n_agents = n_agents
        self.agents = [
            DQNAgent(
                state_size=state_size,
                action_size=15,  # 5 move x 3 pick/drop
                gamma=gamma,
                epsilon=epsilon,
                epsilon_min=epsilon_min,
                epsilon_decay=epsilon_decay,
                learning_rate=learning_rate,
                batch_size=batch_size,
                memory_capacity=memory_capacity
            ) for _ in range(n_agents)
        ]

    def select_actions(self, state_flat):
        # state_flat: state toàn cục, mỗi agent đều nhận được
        actions = []
        for agent in self.agents:
            action = agent.select_action(state_flat)
            actions.append(action)
        return actions

    def store_transitions(self, state_flat, actions, next_state_flat, reward, done):
        for i, agent in enumerate(self.agents):
            agent.store_transition(state_flat, actions[i], next_state_flat, reward, done)

    def optimize_models(self):
        for agent in self.agents:
            agent.optimize_model()

    def update_target_networks(self):
        for agent in self.agents:
            agent.update_target_network()

# --- Simple training loop for map1.txt ---
if __name__ == "__main__":
    env = DeliveryEnv(map_file='map1.txt', n_robots=2, n_packages=2, max_time_steps=30)
    grid_size = env.n_rows * env.n_cols
    robot_size = env.n_robots * 3
    package_size = env.n_packages * 7
    state_size = grid_size + robot_size + package_size
    multi_agent = MultiAgentDQN(
        n_agents=env.n_robots,
        state_size=state_size,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.998,
        learning_rate=1e-4,
        batch_size=64,
        memory_capacity=2000
    )
    num_episodes = 100
    max_steps = 30
    for episode in range(num_episodes):
        state = env.reset()
        grid_flat = state['grid'].flatten()
        robots_flat = state['robots'].flatten()
        packages_flat = state['packages'].flatten()
        state_flat = np.concatenate([grid_flat, robots_flat, packages_flat])
        episode_reward = 0
        done = False
        step = 0
        while not done and step < max_steps:
            actions = multi_agent.select_actions(state_flat)
            env_actions = []
            for a in actions:
                move_action = a // 3
                pkg_action = a % 3
                env_actions.extend([move_action, pkg_action])
            next_state, reward, done, info = env.step(env_actions)
            next_grid_flat = next_state['grid'].flatten()
            next_robots_flat = next_state['robots'].flatten()
            next_packages_flat = next_state['packages'].flatten()
            next_state_flat = np.concatenate([next_grid_flat, next_robots_flat, next_packages_flat])
            multi_agent.store_transitions(state_flat, actions, next_state_flat, reward, done)
            multi_agent.optimize_models()
            state_flat = next_state_flat
            episode_reward += reward
            step += 1
        if episode % 10 == 0:
            multi_agent.update_target_networks()
        print(f"Episode {episode}: Reward={episode_reward:.2f}, Delivery Rate={info['delivery_rate']:.2f}")