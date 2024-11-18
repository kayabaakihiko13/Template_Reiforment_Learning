import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
# QAgent class with basic Q-learning functionality

class QAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        QAgent is a basic reinforcement learning algorithm designed to work in a 2D grid environment,
        where the agent learns optimal actions to maximize cumulative rewards.
        
        Parameters:
        - state_size: Dimension of the state space (assumes a 2D grid).
        - action_size: Number of possible actions.
        - alpha: Learning rate.
        - gamma: Discount factor.
        - epsilon: Exploration rate for epsilon-greedy strategy.
        """
        self.q_table = np.zeros((state_size, state_size, action_size))  # Assuming a 2D grid state
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = action_size
        self.reward_in_episode = []
        self.episode_lengths = []
        self.training_errors = []
        self.episode_errors = []
    def choose_action(self, state):
        """
        Select an action based on the epsilon-greedy strategy.
        
        Parameters:
        - state: Current state as a tuple (row, col).
        
        Returns:
        - The index of the chosen action.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size)  # Explore
        else:
            return np.argmax(self.q_table[state[0], state[1], :])  # Exploit

    def learn(self, state, action_idx, reward, next_state):
        """
        Update the Q-table using the Q-learning algorithm.
        
        Parameters:
        - state: Current state as a tuple (row, col).
        - action_idx: Index of the action taken.
        - reward: Reward received after taking the action.
        - next_state: State reached after taking the action.
        """
        best_next_action = np.max(self.q_table[next_state[0], next_state[1], :])
        td_target = reward + self.gamma * best_next_action
        td_error = td_target - self.q_table[state[0], state[1], action_idx]
        self.q_table[state[0], state[1], action_idx] += self.alpha * td_error
        return td_error
    def update_policy(self, env, num_episodes, verbose=False):
        """
        Train the agent for a specified number of episodes.
        
        Parameters:
        - env: The environment in which the agent operates.
        - num_episodes: Number of episodes to train the agent.
        - verbose: If True, print rewards during training.
        """
        for _ in range(num_episodes):
            state = env.start
            done = False
            total_reward ,steps= 0,0
            
            while not done:
                action_idx = self.choose_action(state)
                action = env.actions[action_idx]
                next_state = env.get_next_state(state, action)
                reward = env.get_reward(next_state)

                # Q-learning update
                td_error = self.learn(state, action_idx, reward, next_state)
                self.episode_errors.append(abs(td_error))

                state = next_state
                total_reward += reward
                steps += 1

                # End episode if goal or trap is reached
                if state == env.goal or state == env.trap:
                    done = True
            self.reward_in_episode.append(total_reward)
            self.episode_lengths.append(steps)
            self.training_errors.append(np.mean(self.episode_errors))
        if verbose:
            print(f"total reward received is {total_reward}")
    def test_policy(self, env):
        """
        Test the agent's learned policy by following the optimal actions.
        
        Parameters:
        - env: The environment in which the agent operates.
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
        elif state == env.trap:
            print("Agent fell into the trap!")
    def plot_result(self) -> None:
        """
        Visualisasikan hasil pelatihan agent, termasuk reward per episode, 
        panjang episode, dan error rata-rata per episode.
        """
        if not hasattr(self, "reward_in_episode") or not hasattr(self, "episode_lengths") or not hasattr(self, "training_errors"):
            print("Error: Anda harus menjalankan update_policy() terlebih dahulu untuk menghasilkan metrik pelatihan.")
            return

        max_episodes_to_plot = 100  # Maksimal jumlah episode yang akan divisualisasikan
        if len(self.reward_in_episode) > max_episodes_to_plot:
            print(f"Memvisualisasikan hanya {max_episodes_to_plot} episode pertama.")
            episodes = range(1, max_episodes_to_plot + 1)
            rewards = self.reward_in_episode[:max_episodes_to_plot]
            lengths = self.episode_lengths[:max_episodes_to_plot]
            errors = self.training_errors[:max_episodes_to_plot]
        else:
            episodes = range(1, len(self.reward_in_episode) + 1)
            rewards = self.reward_in_episode
            lengths = self.episode_lengths
            errors = self.training_errors

        # Buat subplots untuk memvisualisasikan metrik
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))

        # Plot reward per episode
        axes[0].plot(episodes, rewards, label="Total Reward", color="blue")
        axes[0].set_title("Total Reward per Episode")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].grid()
        axes[0].set_ylim([0, max(rewards) * 1.1])  # Batasan sumbu y
        axes[0].legend()

        # Plot panjang episode per episode
        axes[1].plot(episodes, lengths, label="Episode Length", color="green")
        axes[1].set_title("Episode Length per Episode")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Steps")
        axes[1].grid()
        axes[1].set_ylim([0, max(lengths) * 1.1])  # Batasan sumbu y
        axes[1].legend()

        # Plot error rata-rata per episode
        axes[2].plot(episodes, errors, label="Training Error", color="red")
        axes[2].set_title("Average Training Error per Episode")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Error")
        axes[2].grid()
        axes[2].set_ylim([0, max(errors) * 1.1])  # Batasan sumbu y
        axes[2].legend()

        # Atur tata letak plot agar lebih rapi
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)  # Menambahkan jarak antar subplots
        plt.show()



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
