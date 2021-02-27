import sys
import gym
import torch
import pylab
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import random
import numpy as np


# Sum Tree (A binary classification tree where the value of the parent is the sum of its children nodes)
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0

    # Update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on the leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # Store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1

        if self.write >= self.capacity:
            self.write = 0

        if self.write < self.capacity:
            self.n_entries += 1

    # Update priority and sample
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # Get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIDX = idx - self.capacity - 1
        return idx, self.tree[idx], self.data[dataIDX]


# Create a PER Memory
class Memory:
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) * self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        indices = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i+1)

            s = random.uniform(a, b)
            (index, p, data) = self.tree.get()
            priorities.append(p)
            indices.append(index)
            batch.append(data)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, indices, is_weight

    def update(self, index, error):
        p = self._get_priority(error)
        self.tree.update(index, p)


EPISODES = 500


# Approximate Q-Function using a neural network
# Takes in state ans outputs Q value of each actions
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_size)
        )

    def forward(self, x):
        return self.fc(x)


# Create a DQN Agent for the cart pole, it uses a Neural Network to approximate q function
# And it also uses prioritized experience replay memory and target q-network
class DQNAgent():
    def __init__(self, state_size, action_size):
        # Change it to true if you want to see cart pole agent learning
        self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # Hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.memory_size = 20000
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 5000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) /self.explore_step
        self.batch_size = 64
        self.train_start = 1000

        # Create PER from SUMTREE
        self.memory = Memory(self.memory_size)

        # Create main model and target model
        self.model = DQN(state_size, action_size)
        self.model.apply(self.weights_init)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Initialize the target model
        self.update_target_model()

        if self.load_model:
            self.model = torch.load('save_model/cartpole_dqn')

    # Weight xavier initialize
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)

    # Target model update
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Get actions from model using epsilon greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state)
            state = Variable(state).float().cpu()
            q_value = self.model(state)
            _, action = torch.max(q_value, 1)
            return int(action)

    # Save sample to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        target = self.model(Variable(torch.FloatTensor(next_state))).data
        old_value = target[0][action]
        target_value = self.target_model(Variable(torch.FloatTensor(next_state))).data
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.discount_factor * torch.max(target_value)

        error = abs(old_value - target[0][action])
        self.memory.add(error, (state, action, reward, next_state, done))

    # Pick samples from PER Memory in batches
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch, indices, is_weights = self.memory.sample(self.batch_size)
        mini_batch = np.array(mini_batch).transpose()

        states = np.vstack(mini_batch[0])
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = list(mini_batch[3])
        dones = mini_batch[4]

        # Boolean to binary
        dones = dones.astype(int)

        # Q-function of current states
        states = torch.Tensor(states)
        states = Variable(states).float()
        prediction = self.model(states)

        # One-hot encoding
        a = torch.LongTensor(actions).view(-1, 1)

        one_hot_action = torch.FloatTensor(self.batch_size, self.action_size).zero_()
        one_hot_action.scatter_(1, a, 1)

        prediction = torch.sum(prediction.mul(Variable(one_hot_action)), dim=1)

        # Q-function of the next state
        next_states = torch.Tensor(next_states)
        next_states = Variable(next_states).float()
        next_prediction = self.target_model(next_states).data

        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Q-Learning: Get maximum Q value at s' from the target model
        target = rewards + (1 - dones) * self.discount_factor * next_prediction.max(1)[0]
        target = Variable(target)

        errors = torch.abs(prediction - target).data.numpy()

        # Update the priority list
        for i in range(self.batch_size):
            index = indices[i]
            self.memory.update(index, errors[i])

        self.optimizer.zero_grad()

        # MSE Loss function
        loss = (torch.FloatTensor(is_weights) * F.mse_loss(prediction, target).mean())
        loss.backward()

        # Train
        self.optimizer.step()


if __name__ == "__main__":

    # In case of cart pole v1 the episode length is 500
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model = DQN(state_size, action_size)

    agent = DQNAgent(state_size, action_size)
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # Get action for the current state and take one step in the environment
            action = agent.get_action(state)

            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # If an action makes the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -10

            # Save the observation to the replay memory
            agent.append_sample(state, action, reward, next_state, done)

            # Train at every time step
            if agent.memory.tree.n_entries >= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state

            if done:
                # After every episode update the target model to be same with the main model
                agent.update_target_model()

                # After every episode plot the play time
                score = score if score == 500 else + 10
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_dqn.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      agent.memory.tree.n_entries, "  epsilon:", agent.epsilon)

                # If the mean score of the last 10 episodes is greater than 490, then stop the training
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    torch.save(agent.model, './save_model/cart_pole_dqn')
                    sys.exit()
