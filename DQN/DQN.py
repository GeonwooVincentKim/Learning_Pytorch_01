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


# Hyper-Parameter
EPISODES = 50    # Number of Iterations Number
EPS_START = 0.9  # The probability when Agent start to train randomly.
EPS_END = 0.05   # The probability when Agent finish(terminate) to train randomly.
GAMMA = 0.8      # Discount Factor
LR = 0.0001      # Learning Rate
BATCH_SIZE = 64  # Batch-Size
