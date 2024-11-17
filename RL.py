import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# QAgent class with basic Q-learning functionality
class QAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Q Agent merupakan algoritma dasar pada reiforment learning yang dirancang untuk bekerja 
        pada lingkungan grid 2d dengan berbagai aksi yang dapat dilakukan setiap saat

        """
        self.q_table = np.zeros((state_size, state_size, action_size))  # Assuming a 2D grid state
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = action_size

    def choose_action(self, state):
        """
        choose action merupakan statergi aksi berdasarkan nilai epsilon-gredy
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size)  # Explore
        else:
            return np.argmax(self.q_table[state[0], state[1], :])  # Exploit

    def learn(self, state, action_idx, reward, next_state):
        """
        algoritma ini merupakan pembaruan Q table dengan menggunakan algoritma
        Q-Learning
        Formula:
        Q(s,a) <- Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)] 
        langkah algoritma:
        1. Hitung nilai aksi terbaik di keadaan berikutnya (max_a' Q(s', a')).
        2. Hitung target TD (r + γ * max_a' Q(s', a')).
        3. Hitung TD Error (selisih antara TD target dan Q(s, a)).
        4. Perbarui nilai Q-Table untuk keadaan dan aksi tersebut:
        Q(s, a) ← Q(s, a) + α * TD Error
        """
        best_next_action = np.max(self.q_table[next_state[0], next_state[1], :])
        td_target = reward + self.gamma * best_next_action
        td_error = td_target - self.q_table[state[0], state[1], action_idx]
        self.q_table[state[0], state[1], action_idx] += self.alpha * td_error

    def update_policy(self, env, num_episodes):
        """
        memperbarui kebijakan (ekplotasi) agent untuk beberapa episode,dimana setiap
        episode dimulai dari keadaan awal dan berakhir saat agen mencapai tujuan atau
        jabatan
        """
        for _ in range(num_episodes):
            state = env.start
            done = False
            
            while not done:
                action_idx = self.choose_action(state)
                action = env.actions[action_idx]
                next_state = env.get_next_state(state, action)
                reward = env.get_reward(next_state)
                
                # Q-learning update
                self.learn(state, action_idx, reward, next_state)
                state = next_state
                
                # End episode if goal or trap is reached
                if state == env.goal or state == env.trap:
                    done = True

    def test_policy(self, env):
        """
        menggunakan Q table yang udah didapat,agent dapat menjalakan aksi terbaik bedasarkan
        nilai Q yang telah didapat
        """
        state = env.start
        steps = 0
        while state != env.goal and state != env.trap:
            action_idx = np.argmax(self.q_table[state[0], state[1], :])
            action = env.actions[action_idx]
            state = env.get_next_state(state, action)
            print(f"Step {steps}: Agent moved {env.action_names[action_idx]} to {state}")
            steps += 1

        if state == env.goal:
            print("Agent reached the goal!")
        else:
            print("Agent fell into the trap!")

# SARSA Agent that inherits from QAgent
class SARSA(QAgent):
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(state_size, action_size, alpha, gamma, epsilon)

    def learn(self, state, action_idx, reward, next_state, next_action_idx):
        """SARSA update rule"""
        td_target = reward + self.gamma * self.q_table[next_state[0], next_state[1], next_action_idx]
        td_error = td_target - self.q_table[state[0], state[1], action_idx]
        self.q_table[state[0], state[1], action_idx] += self.alpha * td_error

    def update_policy(self, env, num_episodes):
        """SARSA training method"""
        for _ in range(num_episodes):
            state = env.start
            action_idx = self.choose_action(state)
            done = False
            
            while not done:
                action = env.actions[action_idx]
                next_state = env.get_next_state(state, action)
                reward = env.get_reward(next_state)

                # Choose the next action based on SARSA
                next_action_idx = self.choose_action(next_state)

                # Update using SARSA's rule
                self.learn(state, action_idx, reward, next_state, next_action_idx)

                # Move to the next state and action
                state = next_state
                action_idx = next_action_idx
                
                # End episode if goal or trap is reached
                if state == env.goal or state == env.trap:
                    done = True

# Deep Q-Network (DQN) class for approximating Q-values with neural networks
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)  # state_size should match the input features
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent for deep Q-learning with experience replay
class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, epsilon=0.1, device="cpu", buffer_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device

        # Initialize the Q-network and optimizer
        self.model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

    def store_experience(self, state, action, reward, next_state, done):
        """Store an experience in the replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        """Choose an action using epsilon-greedy strategy"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)  # Explore (choose random action)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).view(1, -1)  # Flatten the state
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()  # Exploit (choose action with highest Q-value)

    def update_policy(self, env, num_episodes, batch_size=64):
        """Train the DQN agent by updating the policy using experiences from the replay buffer"""
        for episode in range(num_episodes):
            state = env.reset()  # Reset environment and get initial state
            done = False
            
            while not done:
                action_idx = self.choose_action(state)  # Get action index
                action = env.actions[action_idx]  # Convert index to action tuple
                
                # Take a step in the environment
                next_state, reward, done, _ = env.step(action)
                
                # Store experience in the buffer
                self.store_experience(state, action, reward, next_state, done)
                
                # Sample a batch of experiences from the buffer
                if len(self.replay_buffer) >= batch_size:
                    batch = np.random.choice(len(self.replay_buffer), batch_size)
                    states, actions, rewards, next_states, dones = zip(*[self.replay_buffer[idx] for idx in batch])

                    # Convert to tensors
                    states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
                    actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)  # Ensure this is long tensor
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                    next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(self.device)
                    dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

                    # Q-learning update
                    q_values = self.model(states_tensor)  # Shape: (batch_size, action_size)
                    next_q_values = self.model(next_states_tensor)

                    # Get the Q-value for the taken actions
                    # actions_tensor should be of shape (batch_size,) to index into q_values
                    q_value = q_values.gather(1, actions_tensor.view(-1, 1))  # Ensure actions_tensor has shape (batch_size, 1)
                    q_value = q_value.squeeze(1)  # Now q_value has shape (batch_size,)

                    # Compute the target Q-value (Bellman equation)
                    target_q_value = rewards_tensor + self.gamma * next_q_values.max(1)[0] * (1 - dones_tensor)

                    # Compute the loss
                    loss = self.criterion(q_value, target_q_value)

                    # Optimize the model
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                # Update the state
                state = next_state
