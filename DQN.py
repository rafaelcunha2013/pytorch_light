# from pyvirtualdisplay import Display
# Display(visible=False, size=(1400, 900)).start()

import copy
import gym
import torch
import random

import numpy as np
import torch.nn.functional as F

from collections import deque, namedtuple
# from IPython.display import HTML
# from base64 import b64encode

from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW

from pytorch_lightning import LightningModule, Trainer

from gym.wrappers import RecordVideo, RecordEpisodeStatistics, TimeLimit


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()


class DQN(nn.Module):
    def __init__(self, hidden_size, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x.float())


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


class RLDataset(IterableDataset):
    def __init__(self, buffer, sample_size=200):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        for experience in self.buffer.sample(self.sample_size):
            yield experience


# fn: state --> action
def epsilon_greedy(state, env, net, epsilon=0.0):
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        state = torch.tensor([state]).to(device)
        q_values = net(state)
        _, action = torch.max(q_values, dim=1)
        action = int(action.item())
    return action


def create_environment(name):
    env = gym.make(name)
    env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda x: x % 50 == 0)
    env = RecordEpisodeStatistics(env)
    return env


class DeepQLearning(LightningModule):

    # Initialize.
    def __init__(self, env_name, policy=epsilon_greedy, capacity=100_000, batch_size=254,
               lr=1e-3, hidden_size=128, gamma=0.99, loss_fn=F.smooth_l1_loss, optim=AdamW,
               eps_start=1.0, eps_end=0.15, eps_last_episode=100, samples_per_epoch=10_000, sync_rate=10):
        super().__init__()
        self.env = create_environment(env_name)


        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.q_net = DQN(hidden_size, obs_size, n_actions)

        self.target = copy.deepcopy(self.q_net)
        self.policy = policy
        self.buffer = ReplayBuffer(capacity=capacity)
        self.save_hyperparameters()

        while len(self.buffer) < self.hparams.samples_per_epoch:
            print(f'{len(self.buffer)} samples in experience buffer. Filling...')
            self.play_episode(epsilon=self.hparams.eps_start)

    @torch.no_grad()
    def play_episode(self, policy=None, epsilon=0.0):
        state = self.env.reset()
        done = False

        while not done:
            if policy:
                action = policy(state, self.env, self.q_net, epsilon=epsilon)
            else:
                action = self.env.action_space.sample()
            next_state, reward, done, info = self.env.step(action)
            exp = (state, action, reward, done, next_state)
            self.buffer.append(exp)
            state = next_state
    # Forward.

    # Configure the optimizer.

    # Create dataloader.

    # Training step.

    # Training epoch end.

