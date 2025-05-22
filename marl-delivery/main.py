import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import time
import sys
from contextlib import redirect_stdout
import os
from env import Environment
from greedyagent import GreedyAgents
from dqn_agent import DQNAgent
import argparse
def train(env, agent, num_episodes=200, max_steps=50, target_update=10, state_size=None, bfs_prob=0.3, 
          min_robots=1, max_robots=5, min_packages=2, max_packages=10):
    episode_rewards = []
    delivery_rates = []
    total_distances = []
    pickups_completed = []
    deliveries_completed = []
    best_reward = float('-inf')
    
    greedy_agents = GreedyAgents()
    
    for episode in range(num_episodes):
        n_robots = random.randint(min_robots, max_robots)
        n_packages = random.randint(min_packages, max_packages)
        
        env.n_robots = n_robots
        env.n_packages = n_packages
        
        state = env.reset()
        greedy_agents.init_agents(state)
        
        grid_flat = np.array(state['map']).flatten()
        
        max_robots_local = max_robots
        max_packages_local = max_packages
        
        padded_robots = np.zeros((max_robots_local, 3), dtype=np.int32)
        for i, robot in enumerate(state['robots']):
            if i < max_robots_local:
                padded_robots[i] = robot
        robots_flat = padded_robots.flatten()
        
        packages_arr = np.zeros((max_packages_local, 7), dtype=np.int32)
        for i, p in enumerate(state['packages']):
            if i < max_packages_local:
                packages_arr[i] = p
        packages_flat = packages_arr.flatten()
        
        state_flat = np.concatenate([grid_flat, robots_flat, packages_flat])
      
        assert state_flat.shape[0] == state_size, \
            f"State size mismatch: expected {state_size}, got {state_flat.shape[0]}"
        
        episode_reward = 0
        done = False
        step = 0
        
        episode_total_distance = 0
        episode_pickup_count = 0
        episode_delivery_count = 0
        
        robot_carrying = [False] * n_robots
        robot_positions = [(r[0]-1, r[1]-1) for r in state['robots']]
        
        while not done and step < max_steps:
            greedy_actions = greedy_agents.get_actions(state)
            
            bfs_actions = []
            for move, pkg in greedy_actions:
                move_idx = {'S':0, 'L':1, 'R':2, 'U':3, 'D':4}[move]
                pkg_idx = int(pkg)
                bfs_actions.append((move_idx, pkg_idx))
            
            actions = []
            for i in range(n_robots):
                bfs_action = bfs_actions[i] if i < len(bfs_actions) else None
                action = agent.select_action(state_flat, bfs_action=bfs_action, bfs_prob=bfs_prob)
                actions.append(action)
            
            robot_actions = []
            for i, action in enumerate(actions):
                move_action, pkg_action = action
                move = ['S', 'L', 'R', 'U', 'D'][move_action]
                pkg = str(pkg_action)
                robot_actions.append((move, pkg))
            
            for i, action in enumerate(robot_actions):
                move, pkg = action
            
            prev_positions = robot_positions.copy()
            prev_carrying = robot_carrying.copy()
            
            next_state, reward, done, info = env.step(robot_actions)
            
            robot_positions = [(r[0]-1, r[1]-1) for r in next_state['robots']]
            for i, robot in enumerate(next_state['robots']):
                if robot[2] > 0 and not robot_carrying[i]:
                    robot_carrying[i] = True
                    episode_pickup_count += 1
                elif robot[2] == 0 and robot_carrying[i]:
                    robot_carrying[i] = False
                    episode_delivery_count += 1
            
            step_distance = sum(
                abs(prev[0] - curr[0]) + abs(prev[1] - curr[1])
                for prev, curr in zip(prev_positions, robot_positions)
            )
            episode_total_distance += step_distance
            
            grid_flat = np.array(next_state['map']).flatten()
        
            padded_robots = np.zeros((max_robots_local, 3), dtype=np.int32)
            for i, robot in enumerate(next_state['robots']):
                if i < max_robots_local:
                    padded_robots[i] = robot
            robots_flat = padded_robots.flatten()
            
            packages_arr = np.zeros((max_packages_local, 7), dtype=np.int32)
            for i, p in enumerate(next_state['packages']):
                if i < max_packages_local:
                    packages_arr[i] = p
            packages_flat = packages_arr.flatten()
            
            next_state_flat = np.concatenate([grid_flat, robots_flat, packages_flat])
       
            assert next_state_flat.shape[0] == state_size, \
                f"State size mismatch: expected {state_size}, got {next_state_flat.shape[0]}"
            
            # Store transitions for each agent
            for i, action in enumerate(actions):
                agent.store_transition(state_flat, action, next_state_flat, reward, done)
            
            try:
                agent.optimize_model()
            except Exception as e:
                # Removed print statement to suppress episode output
                if len(agent.memory) >= agent.batch_size:
                    sample = random.sample(agent.memory, agent.batch_size)
                raise e
            
            state = next_state
            state_flat = next_state_flat
            episode_reward += reward
            step += 1
        
        if episode % target_update == 0:
            agent.update_target_network()
        
        # Calculate metrics
        delivery_rate = episode_delivery_count / max(1, n_packages)
        
        episode_rewards.append(episode_reward)
        delivery_rates.append(delivery_rate)
        total_distances.append(episode_total_distance)
        pickups_completed.append(episode_pickup_count)
        deliveries_completed.append(episode_delivery_count)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.movement_net.state_dict(), 'best_movement_model.pth')
            torch.save(agent.package_net.state_dict(), 'best_package_model.pth')
    
    return {
        'episode_rewards': episode_rewards,
        'delivery_rates': delivery_rates,
        'total_distances': total_distances,
        'pickups_completed': pickups_completed,
        'deliveries_completed': deliveries_completed
    }
def plot_metrics(metrics, map_name):
    plt.figure(figsize=(15, 12))
    plt.subplot(3, 2, 1)
    plt.plot(metrics['episode_rewards'])
    plt.title(f'Episode Rewards ({map_name})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(3, 2, 2)
    plt.plot(metrics['delivery_rates'])
    plt.title(f'Delivery Rates ({map_name})')
    plt.xlabel('Episode')
    plt.ylabel('Rate')
    
    plt.subplot(3, 2, 3)
    plt.plot(metrics['total_distances'])
    plt.title(f'Total Distances ({map_name})')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    
    plt.subplot(3, 2, 4)
    plt.plot(metrics['pickups_completed'])
    plt.title(f'Pickups Completed ({map_name})')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    
    plt.subplot(3, 2, 5)
    plt.plot(metrics['deliveries_completed'])
    plt.title(f'Deliveries Completed ({map_name})')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'training_metrics_{map_name}.png')
    plt.close()

def main(map_file='map.txt', max_robots=5, max_packages=10, max_time_steps=50):
    env = Environment(
        map_file=map_file,
        max_time_steps=max_time_steps,
        n_robots=max_robots,
        n_packages=max_packages,
        move_cost=-0.01,        
        delivery_reward=10.0,   
        delay_reward=1.0        
    )
    
    grid_size = env.n_rows * env.n_cols
    robot_size = max_robots * 3  
    package_size = max_packages * 7  
    state_size = grid_size + robot_size + package_size
    action_size = 15  
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.998,
        learning_rate=1e-4,
        batch_size=128,
        memory_capacity=10000
    )
    
    dummy_state = np.random.rand(state_size)
    dummy_state = torch.FloatTensor(dummy_state).unsqueeze(0).to(agent.device)

    num_episodes = 500
    max_steps = 50
    target_update = 5
    bfs_prob = 0.7 

    min_robots = 1
    min_packages = 2
    
    print("Starting training...")
    start_time = time.time()
    
    # Redirect stdout to suppress episode-level output
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        metrics = train(
            env=env,
            agent=agent,
            num_episodes=num_episodes,
            max_steps=max_steps,
            target_update=target_update,
            state_size=state_size,
            bfs_prob=bfs_prob,
            min_robots=min_robots,
            max_robots=max_robots,
            min_packages=min_packages,
            max_packages=max_packages
        )
    
    end_time = time.time()
    
    # Print only the training summary
    print("\nTraining Summary:")
    print(f"Total training time: {end_time - start_time:.2f} seconds")
    print(f"Average reward: {np.mean(metrics['episode_rewards']):.2f}")
    print(f"Best reward: {np.max(metrics['episode_rewards']):.2f}")
    print(f"Final delivery rate: {metrics['delivery_rates'][-1]:.2f}")
    print(f"Total pickups: {sum(metrics['pickups_completed'])}")
    print(f"Total deliveries: {sum(metrics['deliveries_completed'])}")
    
    map_name = map_file.replace('.txt', '')
    plot_metrics(metrics, map_name)
    print(f"\nTraining metrics saved to 'training_metrics_{map_name}.png'")

if __name__ == "__main__":
    if __name__ == "__main__":
    # Set up argument parser
        parser = argparse.ArgumentParser(description="Run DQN training with specified parameters")
        parser.add_argument("--seed", type=int, default=10, help="Random seed for reproducibility")
        parser.add_argument("--max_time_steps", type=int, default=1000, help="Maximum time steps per episode")
        parser.add_argument("--map", type=str, default="map.txt", help="Map file to use")
        parser.add_argument("--num_agents", type=int, default=5, help="Number of robot agents")
        parser.add_argument("--n_packages", type=int, default=10, help="Number of packages")

        # Parse arguments
        args = parser.parse_args()

        # Set random seed for reproducibility
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        main(
            map_file=args.map,
            max_robots=args.num_agents,
            max_packages=args.n_packages,
            max_time_steps=args.max_time_steps
        )