from DQN import *
import numpy as np
from os import path


class Agent:
    def __init__(self, gamma=0.99, epsilon=1.0, lr=0.03, input_dims=[9], batch_size=64, n_actions=2, max_mem_size=1000000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DQN(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)

        if path.exists("data/states.txt"):
            self.state_memory = np.loadtxt("data/states.txt", dtype=np.int32).reshape(self.mem_size, *input_dims)
        else:
            self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.int32)
        if path.exists("data/new_states.txt"):
            self.new_state_memory = np.loadtxt("data/new_states.txt", dtype=np.int32).reshape(self.mem_size, *input_dims)
        else:
            self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.int32)
        if path.exists("data/actions.txt"):
            self.action_memory = np.loadtxt("data/actions.txt", dtype=np.int32).reshape(self.mem_size, 1)
        else:
            self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        if path.exists("data/rewards.txt"):
            self.reward_memory = np.loadtxt("data/rewards.txt", dtype=np.float32).reshape(self.mem_size, 1)
        else:
            self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        #self.waiting_time_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transitions(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        states = open("data/states.txt", "a")
        new_states = open("data/new_states.txt", "a")
        actions = open("data/actions.txt", "a")
        rewards = open("data/rewards.txt", "a")

        if index == 0:
            states.truncate(0)
            new_states.truncate(0)
            actions.truncate(0)
            rewards.truncate(0)

        self.state_memory[index] = state
        np.savetxt(states, state)

        self.new_state_memory[index] = state_
        np.savetxt(new_states, state_)

        self.reward_memory[index] = reward
        np.savetxt(rewards, np.array([reward]))

        self.action_memory[index] = action
        np.savetxt(actions, np.array([action]))

        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def predict_action(self, state, conn=None, vehicle_ids=None):
        if np.random.random() > self.epsilon:
            state = T.tensor([state]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def select_action(self, state, conn=None, vehicle_ids=None):
        #if np.random.random() > self.epsilon:
        state = T.tensor([state]).to(self.Q_eval.device)
        actions = self.Q_eval.forward(state)
        action = T.argmax(actions).item()
        #else:
            #action = np.random.choice(self.action_space)

        return action