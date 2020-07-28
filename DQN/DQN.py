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
        
        Therefore, max(1)[1] is bring maximum value digit from Dimension-2.
        For example, if the shape of code like this
        -> torch.Tensor([1, 3, 2]).max(0)
        
        -> torch.return_types.max(
            values=tensor(3.),
            indices=tensor(1)
        )
    """
    def act(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if random.random() > eps_threshold:
            return self.model(state).data.max(1)[1].view(1, 1)
        else:
            return torch.LongTensor([[random.randrange(2)]])

    """
        'learn()' function role in a training procedure
        that Agent replay experience.
    """
    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)

        # We just prepared experience Sample
        # for training, so now the Agent's Neural Network
        # are going on training procedure.
        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)

        # Calculates the value of left-hand
        # and right-hand behavior by passing
        # the current Status through
        # the Neural Network.
        current_q = self.model(states).gather(1, actions)

        """
            - 1.Future-Value is the expected value of reward 
              that Agent can accept in the future.
            
            - 2. Discounting means that you can get 1 reward 
              in the present and that you will value 
              the current reward more when you get 1 
              reward 1 in the future.
        """
        max_next_q = self.model(next_states).detach().max(1)[0]
        expected_q = rewards + (GAMMA * max_next_q)

        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


"""
    Start Training
"""
env = gym.make('CartPole-v0')
agent = DQNAgent()
score_history = []

"""
    - 1. Create state that includes current Game Conditions
         as Tensor, and use 'act()' function, The Agent Action function.
    
    - 2. The agents that receive the status spit out the action 
         according to the Epsilon Gridi algorithm.

    - 3. The Action variable is a Pytorch Tensor.
         Therefore, extract Agent action number by using
         'item()' function and input into function,
         display 'next_state', 'reward', and 'done' by Agent
         actions.
"""
for e in range(1, EPISODES+1):
    state = env.reset()
    steps = 0

    while True:
        env.render()
        state = torch.FloatTensor([state])
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action.item())

        # Rewards Minus when the game is over.
        if done:
            reward = -1

        agent.memorize(state, action, reward, next_state)
        agent.learn()

        state = next_state
        steps += 1

        # Show the game result when the game finished.
        if done:
            print("Episodes: {0} - Score: {1}".format(e, steps))
            score_history.append(steps)
            break


plt.plot(score_history)
plt.ylabel('score')
plt.show()
