"""Minimal DQN: small MLP, replay buffer, target network, epsilon-greedy.

Kept deliberately simple (no CNN, no image input yet) since the current
state is just a visited-tile mask -- see rl_env.py's docstring for why.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s2, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3, gamma: float = 0.95):
        self.action_dim = action_dim
        self.gamma = gamma
        self.q_net = QNet(state_dim, action_dim)
        self.target_net = QNet(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.loss_fn = nn.HuberLoss()

    def select_action(self, state, visited, epsilon: float) -> int:
        valid_actions = [i for i, v in enumerate(visited) if v == 0]
        if random.random() < epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            q_values = self.q_net(torch.tensor(state, dtype=torch.float32))
            q_values = q_values.numpy().copy()
            for i, v in enumerate(visited):
                if v == 1:
                    q_values[i] = -1e9
            return int(np.argmax(q_values))

    def train_step(self, batch_size: int = 32):
        if len(self.buffer) < batch_size:
            return None
        s, a, r, s2, done = self.buffer.sample(batch_size)

        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(s2).max(dim=1).values
            target = r + self.gamma * next_q * (1 - done)

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
