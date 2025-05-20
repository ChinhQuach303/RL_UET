import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import time
from env import Environment
from greedyagent import GreedyAgents
from dqn_agent import DQNAgent

def train(env, agent, num_episodes=100, max_steps=50, target_update=10, state_size=None, bfs_prob=0.3, 
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
        print(f"\n=== Episode {episode} ===")
        print(f"Number of robots: {n_robots}, Number of packages: {n_packages}")
        
        print("\nInitial State Shapes:")
        print(f"Map size: {len(state['map'])}x{len(state['map'][0])}")
        print(f"Robots: {len(state['robots'])}")
        print(f"Packages: {len(state['packages'])}")
        
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
        
        print("\nFlattened Component Sizes:")
        print(f"Grid flat: {grid_flat.shape}")
        print(f"Robots flat: {robots_flat.shape}")
        print(f"Packages flat: {packages_flat.shape}")
        
        state_flat = np.concatenate([grid_flat, robots_flat, packages_flat])
        print(f"\nCombined state shape: {state_flat.shape}")
        print(f"Expected state size: {state_size}")
        
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
            
            print(f"\nStep {step} Actions:")
            for i, action in enumerate(robot_actions):
                move, pkg = action
                print(f"Robot {i}: move={move}, package={pkg}")
            
            prev_positions = robot_positions.copy()
            prev_carrying = robot_carrying.copy()
            

            next_state, reward, done, info = env.step(robot_actions)
            

            robot_positions = [(r[0]-1, r[1]-1) for r in next_state['robots']]
            for i, robot in enumerate(next_state['robots']):
                if robot[2] > 0 and not robot_carrying[i]:
                    
                    robot_carrying[i] = True
                    episode_pickup_count += 1
                    print(f"Robot {i} picked up package {robot[2]}")
                elif robot[2] == 0 and robot_carrying[i]:
                    
                    robot_carrying[i] = False
                    episode_delivery_count += 1
                    print(f"Robot {i} delivered a package")
            
            
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
            
            print(f"\nNext state shapes:")
            print(f"Map size: {len(next_state['map'])}x{len(next_state['map'][0])}")
            print(f"Robots: {len(next_state['robots'])}")
            print(f"Packages: {len(next_state['packages'])}")
            print(f"Combined: {next_state_flat.shape}")
            
            assert next_state_flat.shape[0] == state_size, \
                f"State size mismatch: expected {state_size}, got {next_state_flat.shape[0]}"
            
            # Store transitions for each agent
            for i, action in enumerate(actions):
                agent.store_transition(state_flat, action, next_state_flat, reward, done)
            
            try:
                agent.optimize_model()
            except Exception as e:
                print(f"\nError in optimize_model: {str(e)}")
                print("Current memory size:", len(agent.memory))
                print("Batch size:", agent.batch_size)
                if len(agent.memory) >= agent.batch_size:
                    sample = random.sample(agent.memory, agent.batch_size)
                    print("Sample state shapes:", [s[0].shape for s in sample])
                raise e
            
            state = next_state
            state_flat = next_state_flat
            episode_reward += reward
            step += 1
            
            if step % 10 == 0:
                print(f"\nStep {step} Summary:")
                print(f"Reward: {reward:.2f}")
                print(f"Total Reward: {episode_reward:.2f}")
                print(f"Pickups: {episode_pickup_count}, Deliveries: {episode_delivery_count}")
                print(f"Total Distance: {episode_total_distance}")
        
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
        
        print(f"\nEpisode {episode} Summary:")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Steps: {step}")
        print(f"Robots: {n_robots}, Packages: {n_packages}")
        print(f"Pickups: {episode_pickup_count}, Deliveries: {episode_delivery_count}")
        print(f"Delivery Rate: {delivery_rate:.2f}")
        print(f"Total Distance: {episode_total_distance}")
        print(f"Episode {episode} info: {info}")
        print("-" * 50)
    
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

def main(map_file='map1.txt', max_robots=5, max_packages=10, max_time_steps=50):
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
    
    print(f"Map: {map_file}")
    print(f"Grid size: {grid_size} ({env.n_rows}x{env.n_cols})")
    print(f"Max robots: {max_robots}, Robot state size: {robot_size}")
    print(f"Max packages: {max_packages}, Package state size: {package_size}")
    print(f"Total state size: {state_size}")
    print(f"Action size: {action_size}")
    
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
    print("Movement network test output:", agent.movement_net(dummy_state))
    print("Package network test output:", agent.package_net(dummy_state))
    
    num_episodes = 500
    max_steps = 50
    target_update = 5
    bfs_prob = 0.7 

    min_robots = 1
    min_packages = 2
    
    print("Starting training...")
    start_time = time.time()
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
    main(max_robots=5, max_packages=10)