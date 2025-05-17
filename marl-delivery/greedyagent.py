import numpy as np

def run_bfs(map, start, goal, robot_positions=None):
    """
    Run BFS to find the shortest path from start to goal, considering robot positions to avoid collisions.
    :param map: 2D grid map (0: free, 1: obstacle)
    :param start: Starting position (row, col)
    :param goal: Goal position (row, col)
    :param robot_positions: List of other robot positions to avoid
    :return: (move, distance) - move action and distance to goal
    """
    if robot_positions is None:
        robot_positions = []

    n_rows = len(map)
    n_cols = len(map[0])
    queue = []
    visited = set()
    queue.append((goal, []))
    visited.add(goal)
    d = {}
    d[goal] = 0

    while queue:
        current, path = queue.pop(0)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (current[0] + dx, current[1] + dy)
            if (next_pos[0] < 0 or next_pos[0] >= n_rows or next_pos[1] < 0 or next_pos[1] >= n_cols or
                map[next_pos[0]][next_pos[1]] == 1 or next_pos in robot_positions):
                continue
            if next_pos not in visited:
                visited.add(next_pos)
                d[next_pos] = d[current] + 1
                queue.append((next_pos, path + [next_pos]))

    if start not in d:
        return 'S', 100000

    actions = ['S', 'L', 'R', 'U', 'D']
    directions = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
    current = start
    for i, (dx, dy) in enumerate(directions):
        next_pos = (current[0] + dx, current[1] + dy)
        if next_pos in d and d[next_pos] == d[current] - 1:
            return actions[i], d[next_pos]
    return 'S', d[start]

class GreedyAgents:
    """
    A greedy agent implementation for multi-robot package delivery with improved assignment and pathfinding.
    """
    IDLE = 'idle'
    MOVING_TO_PICKUP = 'to_pickup'
    MOVING_TO_DELIVER = 'to_deliver'

    def __init__(self):
        self.robots = []
        self.robot_states = []
        self.robot_targets = []
        self.robot_last_progress = []  # Track last time robot made progress
        self.packages = []
        self.package_assigned = []
        self.n_robots = 0
        self.map = None
        self.current_time = 0  # Track current timestep
        self.is_initialized = False

    def init_agents(self, state):
        """Initialize agents from environment state."""
        self.n_robots = len(state['robots'])
        self.map = state['map']
        self.current_time = state['time_step']

        self.robots = [(r[0], r[1], r[2]) for r in state['robots']]  # Already 0-based from updated env.py
        self.robot_states = [self.IDLE] * self.n_robots
        self.robot_targets = [-1] * self.n_robots
        self.robot_last_progress = [self.current_time] * self.n_robots

        self.packages = []
        for p in state['packages']:
            self.packages.append((p[0], p[1], p[2], p[3], p[4], p[5], p[6]))  # Already 0-based

        self.package_assigned = [False] * len(self.packages)
        self.is_initialized = True
        print(f"Initialized {self.n_robots} robots and {len(self.packages)} packages")

    def update_state(self, state):
        """Update agent state from environment state and handle reassignments."""
        if not self.is_initialized:
            self.init_agents(state)
            return

        self.current_time = state['time_step']
        old_positions = [(r[0], r[1]) for r in self.robots]

        # Update robot positions and states
        for i, robot in enumerate(state['robots']):
            carrying_pkg_id = robot[2]
            old_pos = (self.robots[i][0], self.robots[i][1])
            new_pos = (robot[0], robot[1])

            self.robots[i] = (new_pos[0], new_pos[1], carrying_pkg_id)

            # Update progress timestamp if robot moved
            if old_pos != new_pos:
                self.robot_last_progress[i] = self.current_time

            # Handle state transitions
            if carrying_pkg_id == 0:
                if self.robot_states[i] == self.MOVING_TO_DELIVER:
                    print(f"Robot {i} delivered package {self.robot_targets[i]}")
                    self.robot_states[i] = self.IDLE
                    pkg_idx = self.robot_targets[i]
                    if 0 <= pkg_idx < len(self.package_assigned):
                        self.package_assigned[pkg_idx] = False
                    self.robot_targets[i] = -1
                    self.robot_last_progress[i] = self.current_time
            else:
                if self.robot_states[i] == self.MOVING_TO_PICKUP:
                    print(f"Robot {i} picked up package {carrying_pkg_id}")
                    self.robot_states[i] = self.MOVING_TO_DELIVER
                    self.robot_targets[i] = carrying_pkg_id - 1
                    self.robot_last_progress[i] = self.current_time

            # Check for timeout (10 steps without progress)
            if self.robot_states[i] in [self.MOVING_TO_PICKUP, self.MOVING_TO_DELIVER]:
                if self.current_time - self.robot_last_progress[i] > 10:
                    print(f"Robot {i} timed out on package {self.robot_targets[i]}, reassigning")
                    self.robot_states[i] = self.IDLE
                    pkg_idx = self.robot_targets[i]
                    if 0 <= pkg_idx < len(self.package_assigned):
                        self.package_assigned[pkg_idx] = False
                    self.robot_targets[i] = -1
                    self.robot_last_progress[i] = self.current_time

        # Update package information
        self.packages = []
        for p in state['packages']:
            self.packages.append((p[0], p[1], p[2], p[3], p[4], p[5], p[6]))

        # Update package assignments
        self.package_assigned = [False] * len(self.packages)
        for i in range(self.n_robots):
            if self.robot_states[i] != self.IDLE and self.robot_targets[i] >= 0:
                pkg_idx = self.robot_targets[i]
                if 0 <= pkg_idx < len(self.package_assigned):
                    self.package_assigned[pkg_idx] = True

    def find_closest_package(self, robot_id, robot_pos):
        """Find the closest unassigned package, considering start_time and deadlines."""
        available_packages = []
        for i, pkg in enumerate(self.packages):
            if not self.package_assigned[i] and pkg[5] <= self.current_time:  # Check start_time
                pkg_pos = (pkg[1], pkg[2])
                dist = abs(pkg_pos[0] - robot_pos[0]) + abs(pkg_pos[1] - robot_pos[1])
                deadline_urgency = max(0, pkg[6] - self.current_time)  # Lower means more urgent
                score = dist - 0.1 * deadline_urgency  # Prioritize closer and more urgent packages
                available_packages.append((i, dist, score))

        if not available_packages:
            return None

        # Sort by combined score (distance adjusted by urgency)
        available_packages.sort(key=lambda x: x[2])

        # Assign to the robot that's the best candidate (closest and lowest ID if tied)
        for pkg_idx, pkg_dist, _ in available_packages:
            is_best_robot = True
            for other_robot_id in range(self.n_robots):
                if other_robot_id == robot_id:
                    continue
                if self.robot_states[other_robot_id] != self.IDLE:
                    continue
                other_robot_pos = (self.robots[other_robot_id][0], self.robots[other_robot_id][1])
                pkg = self.packages[pkg_idx]
                pkg_pos = (pkg[1], pkg[2])
                other_dist = abs(pkg_pos[0] - other_robot_pos[0]) + abs(pkg_pos[1] - other_robot_pos[1])
                if other_dist < pkg_dist and other_robot_id < robot_id:
                    is_best_robot = False
                    break
            if is_best_robot:
                return pkg_idx

        return available_packages[0][0] if available_packages else None

    def get_next_action(self, robot_id):
        """Get the next action for a robot, considering collisions and timing."""
        robot = self.robots[robot_id]
        robot_pos = (robot[0], robot[1])

        # Validate robot position
        if (robot_pos[0] < 0 or robot_pos[0] >= len(self.map) or
            robot_pos[1] < 0 or robot_pos[1] >= len(self.map[0])):
            print(f"Robot {robot_id} at invalid position {robot_pos}, staying")
            return 'S', '0'

        state = self.robot_states[robot_id]

        # Collect positions of other robots for collision avoidance
        other_positions = [(r[0], r[1]) for i, r in enumerate(self.robots) if i != robot_id]

        if state == self.IDLE:
            closest_pkg_idx = self.find_closest_package(robot_id, robot_pos)
            if closest_pkg_idx is not None:
                pkg_id = closest_pkg_idx
                self.robot_targets[robot_id] = pkg_id
                self.robot_states[robot_id] = self.MOVING_TO_PICKUP
                self.package_assigned[pkg_id] = True
                print(f"Robot {robot_id} assigned to package {pkg_id} (ID:{self.packages[pkg_id][0]})")

                pkg = self.packages[pkg_id]
                target_pos = (pkg[1], pkg[2])
                if robot_pos == target_pos:
                    print(f"Robot {robot_id} already at pickup location, picking up package {pkg_id}")
                    return 'S', '1'
                else:
                    move, _ = run_bfs(self.map, robot_pos, target_pos, other_positions)
                    return move, '0'
            else:
                return 'S', '0'

        elif state == self.MOVING_TO_PICKUP:
            pkg_id = self.robot_targets[robot_id]
            if pkg_id < 0 or pkg_id >= len(self.packages):
                print(f"Invalid package ID {pkg_id} for robot {robot_id}, setting to IDLE")
                self.robot_states[robot_id] = self.IDLE
                self.robot_targets[robot_id] = -1
                return 'S', '0'

            pkg = self.packages[pkg_id]
            if pkg[5] > self.current_time:  # Package not yet available
                return 'S', '0'

            target_pos = (pkg[1], pkg[2])
            if robot_pos == target_pos:
                print(f"Robot {robot_id} at pickup location, picking up package {pkg_id}")
                return 'S', '1'
            else:
                move, _ = run_bfs(self.map, robot_pos, target_pos, other_positions)
                return move, '0'

        elif state == self.MOVING_TO_DELIVER:
            pkg_id = self.robot_targets[robot_id]
            if pkg_id < 0 or pkg_id >= len(self.packages):
                print(f"Invalid package ID {pkg_id} for robot {robot_id}, setting to IDLE")
                self.robot_states[robot_id] = self.IDLE
                self.robot_targets[robot_id] = -1
                return 'S', '0'

            pkg = self.packages[pkg_id]
            target_pos = (pkg[3], pkg[4])
            if robot_pos == target_pos:
                print(f"Robot {robot_id} at delivery location, dropping package {pkg_id}")
                return 'S', '2'
            else:
                move, _ = run_bfs(self.map, robot_pos, target_pos, other_positions)
                return move, '0'

        return 'S', '0'

    def get_actions(self, state):
        """Get actions for all robots."""
        self.update_state(state)

        actions = []
        for i in range(self.n_robots):
            move, pkg_action = self.get_next_action(i)
            actions.append((move, pkg_action))

        print("\nRobot States:")
        for i in range(self.n_robots):
            robot = self.robots[i]
            print(f"Robot {i}: pos={robot[0:2]}, carrying={robot[2]}, " +
                  f"state={self.robot_states[i]}, target={self.robot_targets[i]}")

        print("\nPackage Assignments:")
        for i, pkg in enumerate(self.packages):
            status = "Assigned" if self.package_assigned[i] else "Available"
            print(f"Package {i} (ID:{pkg[0]}): {status}")

        print("\nActions:", actions)
        return actions

        