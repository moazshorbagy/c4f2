import numpy as np
from sumo_utils import get_total_waiting_time

class DeepQNetwork:
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

class Agent:

    def __init__(self):
        print("Constructing agent")
        n_actions = 2
        input_dims = 9

        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_dec = 5e-4
        self.lr = 0.0015
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = 500 # 100000
        self.batch_size = 64
        self.mem_counter = 0

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.int32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.int32)

        self.previous_state = np.zeros(input_dims, dtype=np.int32)

    def store_transition(self, state, reward, new_state):
        """ Stores transitions in the agent's memory """
        index = self.mem_counter % self.mem_size # wrap around the mem counter since memory is finite
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        
        self.mem_counter += 1

    def store_memory(self):
        with open("data/state_memory.txt", "w") as state_memory_file:
            for state in self.state_memory:
                state_memory_file.write(np.array2string(state, separator=','))
                state_memory_file.write('\n')
        
        with open("data/new_state_memory.txt", "w") as new_state_memory_file:
            for state in self.new_state_memory:
                new_state_memory_file.write(np.array2string(state, separator=','))
                new_state_memory_file.write('\n')
        
        with open("data/reward_memory.txt", "w") as reward_memory_file:
            for reward in self.reward_memory:
                reward_memory_file.write(np.array2string(reward, separator=','))
                reward_memory_file.write('\n')

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            # predict (i.e. feed forward) and action = argmax(actions)
            action = np.random.choice(self.action_space)
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def calculate_reward(self, conn, vehicle_ids):
        return -1 * get_total_waiting_time(conn, vehicle_ids)

    def learn(self):
        if self.mem_counter < self.batch_size:
            return
        
        # calculate loss and do backward propagation to update weights

    def select_action(self, state, conn=None, vehicle_ids=None):
        if conn is not None:
            reward = self.calculate_reward(conn, vehicle_ids)
            self.store_transition(self.previous_state, reward, state)
            self.previous_state = state

            self.learn()
        return self.choose_action(state)


# play a bunch of games to fill up the memory because we can't learn from zeros (i.e. constructing a dataset)
