"""
    1. What does 'reinforcement learning' stands for.
    - The field of machine learning which grows in the direction
    of getting good grades by interacting with a given environment
    is named as 'Reinforcement-Learning.'
    as 4 elements.

    2. Reinforcement-Learning could be divide as
    - 1. State - AI Player.
    - 2. Agent - It is a stage that Agent trying to find Solution.
    - 3. Action - It is interacting with a given environment
    (or normal Environment not a given environment) that Agent
    enforce in the Environment.
    - 4. Reward - It is a result or score by Agent's action.
"""
# Installing gym by using pip Module.
import gym
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

"""
    1. EPISODES
    --> The number of Playing game.
    2. EPSILON(EPS_START, EPS_END)
    --> Rate of Agent acts randomly.
    3. EPSILON Reduction ratio(EPS_DECAY)
    --> Reducing Value of Epsilon value that
    starting EPS_START and finishing as EPS_END.
    4. GAMMA
    --> Gamma is the value of 
    how the agent values current rewards 
    over future rewards.
"""
# Hyper-Parameter
EPISODES = 50  # Number of Iterations Number
EPS_START = 0.9  # The probability when Agent start to train randomly.
EPS_END = 0.05  # The probability when Agent finish(terminate) to train randomly.
EPS_DECAY = 200  # A value that reduce the probability when Agent proceed to train randomly.
GAMMA = 0.8  # Discount Factor
LR = 0.0001  # Learning Rate
BATCH_SIZE = 64  # Batch-Size

"""
    DQN Agent
"""


class DQNAgent:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.optimizer = optim.Adam(self.model.parameters(), LR)
        self.steps_done = 0

        # Create Matrix(Array) for storage memory.
        # Use Queue Data Structure to implement 'memory Class'.
        self.memory = deque(maxlen=10000)

    def memorize(self, state, action, reward, next_state):
        self.memory.append((
            state, action,
            torch.FloatTensor([reward]),
            torch.FloatTensor([next_state])
        ))

    """
        The Pytorch Tensor of max() function get arguments and 
        convert to Matrix Tensor shape into Max-Value and Min-Value(Index).
    """
    def act(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if random.random() > eps_threshold:
            return self.model(state).data.max(1)[1].view(1, 1)
        else:
            return torch.LongTensor([[random.randrange(2)]])
