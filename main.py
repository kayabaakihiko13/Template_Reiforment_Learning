import numpy as np
import torch
from RL import QAgent, SARSA, DQNAgent

class Environment:
    def __init__(self, grid_size, start, goal, trap):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.trap = trap
        self.actions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # Right, Left, Up, Down
        self.action_names = ["Right", "Left", "Up", "Down"]
        self.state = start  # Initial state is the start position

    def get_next_state(self, state, action):
        """Return the next state after taking an action"""
        row = state[0] + action[0]
        col = state[1] + action[1]
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            return (row, col)
        else:
            return state  # Stay in place if out of bounds

    def reset(self):
        """Reset the environment to the start position"""
        self.state = self.start
        return self.state

    def get_reward(self, state):
        """Return the reward for being in a given state"""
        if state == self.goal:
            return 10  # Reward for reaching the goal
        elif state == self.trap:
            return -10  # Penalty for falling into the trap
        else:
            return -1  # Penalty for each step

    def step(self, action):
        """Take a step in the environment based on the action"""
        next_state = self.get_next_state(self.state, action)
        reward = self.get_reward(next_state)
        done = (next_state == self.goal or next_state == self.trap)  # End episode if goal or trap is reached
        self.state = next_state  # Update the state
        return next_state, reward, done, {}  # Return the next state, reward, and done flag


if __name__ == "__main__":
    # Initialize environment and DQN agent
    grid_size = 5
    env = Environment(grid_size=grid_size, start=(0, 0), goal=(4, 4), trap=(2, 2))

    state_size = 2  # Representing the state as (row, col)
    action_size = len(env.actions)  # Number of actions available

    agent = DQNAgent(state_size=state_size, action_size=action_size, lr=0.001, gamma=0.99, epsilon=0.1, device="cpu")

    # Train the agent using DQN
    num_episodes = 1000
    agent.update_policy(env, num_episodes)

    # Test the learned policy
    agent.test_policy(env)
