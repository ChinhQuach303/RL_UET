import numpy as np

class Robot: 
    def __init__(self, position): 
        self.position = position # vị trí robot
        self.carrying = 0  # có đang mang hàng hay không (có hoặc không)

class Package: 
    def __init__(self, start, start_time, target, deadline, package_id): 
        self.start = start
        self.start_time = start_time
        self.target = target
        self.deadline = deadline
        self.package_id = package_id
        self.status = 'None'  # None, Waiting, In Transit, Delivered

class Environment: 
    def __init__(self, map_file, max_time_steps=100, n_robots=5, n_packages=20,
                 move_cost=-0.01, delivery_reward=10., delay_reward=1., 
                 seed=2025): 
        """ Initializes the simulation environment. 
        :param map_file: map file 
        :param max_time_steps: Maximum number of time steps (T).
        :param n_robots: Number of robots (C).
        :param n_packages: Number of packages to generate.
        :param move_cost: Cost incurred when a robot moves (LRUD). Default: -0.01.
        :param delivery_reward: Reward for delivering a package on time. Default: 10.0.
        :param delay_reward: Reward for delivering a package late. Default: 1.0.
        :param seed: Random seed for reproducibility.
        """ 
        self.map_file = map_file
        self.grid = self.load_map()
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0]) if self.grid else 0 
        self.move_cost = move_cost 
        self.delivery_reward = delivery_reward 
        self.delay_reward = delay_reward
        self.t = 0 
        self.robots = []
        self.packages = []
        self.total_reward = 0
        self.n_robots = n_robots
        self.max_time_steps = max_time_steps
        self.n_packages = n_packages
        self.rng = np.random.RandomState(seed)
        self.reset()
        self.done = False
        self.state = None

    def load_map(self):
        """
        Tạo một grid với đầu vào là map_file
        """
        grid = []
        with open(self.map_file, 'r') as f:
            for line in f:
                row = [int(x) for x in line.strip().split(' ')]
                grid.append(row)
        return grid
    
    def is_free_cell(self, position):
        """
        Kiểm tra xem vị trí đang đứng có hợp lệ không bao gồm điều kiện row, col và grid[row][col] == 0
        """
        r, c = position
        if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
            return False
        return self.grid[r][c] == 0

    def add_robot(self, position):
        """
        Tạo robot tại vị trí hợp lệ
        """
        if self.is_free_cell(position):
            robot = Robot(position)
            self.robots.append(robot)
        else:
            raise ValueError("Invalid robot position: must be on a free cell not occupied by an obstacle or another robot.")

    def reset(self):
        """
        Khởi tạo trạng thái ban đầu 
        """
        self.t = 0
        self.robots = []
        self.packages = []
        self.total_reward = 0
        self.done = False
        self.state = None

        # Tạo grid
        tmp_grid = np.array(self.grid)
        for i in range(self.n_robots):
            # Check position-> add robot
            position, tmp_grid = self.get_random_free_cell(tmp_grid)
            self.add_robot(position)
        
        N = self.n_rows
        list_packages = []
        for i in range(self.n_packages):
            start = self.get_random_free_cell_p()
            while True:
                target = self.get_random_free_cell_p()
                if start != target:
                    break
            to_deadline = 10 + self.rng.randint(N//2, 3*N)
            if i <= min(self.n_robots, 20):
                start_time = 0
            else:
                start_time = self.rng.randint(1, self.max_time_steps)
            list_packages.append((start_time, start, target, start_time + to_deadline))

        list_packages.sort(key=lambda x: x[0])
        for i in range(self.n_packages):
            start_time, start, target, deadline = list_packages[i]
            package_id = i+1
            package = Package(start, start_time, target, deadline, package_id)
            if start_time == 0:
                package.status = 'waiting'
            self.packages.append(package)

        return self.get_state()
    
    def get_state(self):
        """
        Khởi tạo trạng thái
        """
        for i in range(len(self.packages)):
            # Nếu thời gian nhận hàng bằng thời gian khởi tạo môi trường thì mặc định là waiting
            if self.packages[i].start_time == self.t:
                self.packages[i].status = 'waiting'
        # tạo 1 list chứa những package để giao bao gồm 2 trạng thái "waiting" và 'in_transit'
        active_packages = []
        for pkg in self.packages:
            if pkg.status in ['waiting', 'in_transit'] and pkg.start_time <= self.t:
                active_packages.append(pkg)

        state = {
            'time_step': self.t,
            'map': self.grid,
            'robots': [(robot.position[0], robot.position[1], robot.carrying) for robot in self.robots],
            'packages': [(package.package_id, package.start[0], package.start[1], 
                          package.target[0], package.target[1], package.start_time, package.deadline) 
                         for package in active_packages]
        }
        return state
        
    def get_random_free_cell_p(self):
        """
        Tạo một array chứa các vị trí hợp lệ và lấy 1 điểm trong số đó làm vị trí random cho robot
        """
        free_cells = [(i, j) for i in range(self.n_rows) for j in range(self.n_cols) 
                      if self.grid[i][j] == 0]
        i = self.rng.randint(0, len(free_cells))
        return free_cells[i]

    def get_random_free_cell(self, new_grid):
        """
        Tạo một ví trị random trên grid cũ và mới
        """
        free_cells = [(i, j) for i in range(self.n_rows) for j in range(self.n_cols) 
                      if new_grid[i][j] == 0]
        i = self.rng.randint(0, len(free_cells))
        new_grid[free_cells[i][0]][free_cells[i][1]] = 1
        return free_cells[i], new_grid
    
    def step(self, actions):
        """
        Advances the simulation by one timestep.
        :param actions: A list where each element is a tuple (move_action, package_action) for a robot.
            move_action: one of 'S', 'L', 'R', 'U', 'D'.
            package_action: '0' (do nothing), '1' (pickup), or '2' (drop).
        :return: The updated state, reward, done flag, and info dictionary.
        """
        r = 0
        deliveries = 0
        pickups = 0
        if len(actions) != len(self.robots):
            raise ValueError("The number of actions must match the number of robots.")

        # -------- Process Movement --------
        proposed_positions = [] # lưu các vị trí đã đi(hợp lí)
        old_pos = {}
        next_pos = {}
        for i, robot in enumerate(self.robots):  # pos carrying
            move, pkg_act = actions[i] # [lrud] and [0,1,2]
            new_pos = self.compute_new_position(robot.position, move)
            if not self.valid_position(new_pos):
                new_pos = robot.position
            proposed_positions.append(new_pos)
            old_pos[robot.position] = i # lưu rằng robot đã đi qua vị trí đó
            next_pos[new_pos] = i #lưu rằng robot đã đi qua vị trí đó

        moved_robots = [0] * len(self.robots) # lưu các bước di chuyển [lrud] 
        computed_moved = [0] * len(self.robots)# lưu các bước tính toán(có thể đến hoặc không thể đến vì..)
        final_positions = [None] * len(self.robots)# vị trí cuối cùng(sau khi di chuyển)
        occupied = {}# dictionary lưu rằng vị trí đó là của robot nào sau khi robot di chuyển

        # Tạo lượt di chuyển ngẫu nhiên
        robot_indices = list(range(len(self.robots)))
        self.rng.shuffle(robot_indices)

        while True:
            updated = False

            for i in robot_indices:
                # Nếu robot đã được tính toán nước đi
                if computed_moved[i] != 0:
                    continue
                # Khởi tạo ví trí hiện tại
                pos = self.robots[i].position
                # Vị trí tiếp theo
                new_pos = proposed_positions[i]
                can_move = False
                # Nếu new pos chưa từng có trong old pos tức là chưa nằm trên đường di chuyển của robot nào-> được di chuyển
                if new_pos not in old_pos:
                    can_move = True        
                else:
                # kiểm tra xem robot nào có cùng vị trí đã được thăm trước đó
                    j = old_pos[new_pos]
                # nếu 2 robot khác nhau và robot kia chưa được tính toán di chuyển(tức là trong lượt di chuyển thì 2 robot này không cùng đi vào 1 vị trí hoặc hoán đổi vị trí với nhau), vậy thì nếu j trùng i tức là lặp đi lặp lại?
                    if j != i and computed_moved[j] == 0:
                        continue
                    can_move = True

                if can_move:
                    if new_pos not in occupied: # nếu được di chuyển và vị trí mới chưa bị chiếm
                        occupied[new_pos] = i # vị trí này sẽ là của robo
                        final_positions[i] = new_pos
                        computed_moved[i] = 1
                        moved_robots[i] = 1
                        updated = True
                    else:
                        new_pos = pos # nếu được di chuyển nhưng vị trí mới đã bị chiếm, quay lại vị trí cũ
                        occupied[new_pos] = i 
                        final_positions[i] = pos
                        computed_moved[i] = 1
                        moved_robots[i] = 0
                        updated = True

                if updated:
                    break

            if not updated:
                break

        for i in range(len(self.robots)):
            if computed_moved[i] == 0:
                final_positions[i] = self.robots[i].position

        # cập nhật vị trí
        for i, robot in enumerate(self.robots):
            move, pkg_act = actions[i]
            if move in ['L', 'R', 'U', 'D'] and final_positions[i] != robot.position: 
                r += self.move_cost
            robot.position = final_positions[i]

        # Reward Shaping dựa trên mahattan distance, khoảng cách càng gần thì reward sẽ được cộng thêm càng nhiều 
        for i, robot in enumerate(self.robots):
            if robot.carrying == 0:
                for pkg in self.packages:
                    if pkg.status == 'waiting' and pkg.start_time <= self.t:
                        dist = abs(robot.position[0] - pkg.start[0]) + abs(robot.position[1] - pkg.start[1])
                        r += 0.1 / (dist + 1)  # Encourage moving closer to packages
            elif robot.carrying != 0:
                pkg = self.packages[robot.carrying - 1]
                dist = abs(robot.position[0] - pkg.target[0]) + abs(robot.position[1] - pkg.target[1])
                r += 0.1 / (dist + 1)  # Encourage moving closer to targets

        # nếu robot đến cái vị trí có hàng- nếu đang không mang hàng, thỏa mãn các điều kiện như thời gian hợp lệ(start_time<=t), trùng vị trí, và hàng thì chưa được robot nào mang đi-> ok
        picked_packages = set()
        for i, robot in enumerate(self.robots):
            move, pkg_act = actions[i]
            if pkg_act == '1':
                if robot.carrying == 0:
                    for j in range(len(self.packages)):
                        pkg = self.packages[j]
                        if (pkg.status == 'waiting' and pkg.start == robot.position and 
                            pkg.start_time <= self.t and pkg.package_id not in picked_packages):
                            robot.carrying = pkg.package_id
                            pkg.status = 'in_transit'
                            picked_packages.add(pkg.package_id)
                            pickups += 1
                            print(f"Robot {i} picked up package {pkg.package_id}")
                            break

        # Nếu đến cái vị trí trả hàng, đầu tiên là phải có hàng, sau đó kiểm tra các điểu kiện như target có trùng với position hiện tại không, rồi 
        for i, robot in enumerate(self.robots):
            move, pkg_act = actions[i]
            if pkg_act == '2':
                if robot.carrying != 0:
                    package_id = robot.carrying
                    target = self.packages[package_id - 1].target
                    if robot.position == target:
                        pkg = self.packages[package_id - 1]
                        pkg.status = 'delivered'
                        if self.t <= pkg.deadline:
                            r += self.delivery_reward
                        else:
                            r += self.delay_reward
                        robot.carrying = 0
                        deliveries += 1
                        print(f"Robot {i} delivered package {package_id}")

        # Increment timestep
        self.t += 1
        self.total_reward += r

        # Check termination
        done = False
        infos = {}
        if self.check_terminate():
            done = True
            infos['total_reward'] = self.total_reward
            infos['total_time_steps'] = self.t

        infos['deliveries'] = deliveries
        infos['pickups'] = pickups

        return self.get_state(), r, done, infos
    
    def check_terminate(self):
        """
        Xử lí khi t> max_time_steps -> True
        Còn khi hàng đã giao và t< max_time_steps -> False
        """
        if self.t >= self.max_time_steps:
            return True
        for p in self.packages:
            if p.status != 'delivered' and p.start_time <= self.t:
                return False
        return True

    def compute_new_position(self, position, move):
        """
        Di chuyển dựa trên hành động ['S', 'L', 'R', 'U', 'D']
        """
        r, c = position
        if move == 'S':
            return (r, c)
        elif move == 'L':
            return (r, c - 1)
        elif move == 'R':
            return (r, c + 1)
        elif move == 'U':
            return (r - 1, c)
        elif move == 'D':
            return (r + 1, c)
        else:
            return (r, c)

    def valid_position(self, pos):
        """
        Kiểm tra vị trí có hợp lệ hay không
        """
        r, c = pos
        if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
            return False
        if self.grid[r][c] == 1:
            return False
        return True

    def render(self):
        """
        A simple text-based rendering of the map showing obstacles and robot positions.
        """
        grid_copy = [row[:] for row in self.grid]
        
        for i, robot in enumerate(self.robots):
            r, c = robot.position
            if robot.carrying == 0:
                grid_copy[r][c] = f'R{i}'
            else:
                grid_copy[r][c] = f'C{i}'
        
        for p in self.packages:
            if p.status == 'waiting' and p.start_time <= self.t:
                sr, sc = p.start
                grid_copy[sr][sc] = 'P'
            if p.status != 'delivered':
                tr, tc = p.target
                grid_copy[tr][tc] = 'T'
        
        print(f"\nTime step: {self.t}/{self.max_time_steps}")
        print(f"Total reward so far: {self.total_reward:.2f}")
        for row in grid_copy:
            print('\t'.join(str(cell) for cell in row))
        
if __name__=="__main__":
    env = Environment("map1.txt", 10, 2, 5)
    state = env.reset()
    print("Initial State:", state)
    print("Initial State:")
    env.render()

    from greedyagent import GreedyAgents as Agents
    agents = Agents()
    agents.init_agents(state)
    print("Agents initialized.")
    
    list_actions = ['S', 'L', 'R', 'U', 'D']
    n_robots = len(state['robots'])
    done = False
    t = 0
    while not done:
        actions = agents.get_actions(state) 
        state, reward, done, infos = env.step(actions)
    
        print("\nState after step:")
        env.render()
        print(f"Reward: {reward}, Done: {done}, Infos: {infos}")
        print("Total Reward:", env.total_reward)
        print("Time step:", env.t)
        print("Packages:", state['packages'])
        print("Robots:", state['robots'])

        t += 1
        if t == 100:
            break