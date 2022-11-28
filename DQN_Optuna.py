import torch
from torch import nn
from torch.utils.data.dataset import IterableDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping


import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, TimeLimit

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import random
import numpy as np
from collections import deque
import copy

"""
Code a vanila DQN implementation

Main classes:
-- Model / Qnetwork
        __init__
        forward
-- Replay Buffer
        __init__
        add / push / append
        sample
        __len__
-- Agent
        __init__
        step
        act
        learn / training
"""

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()


class DQN(nn.Module):

    def __init__(self, hidden_dim, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x.float())


class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class RLDataset(IterableDataset):
    def __init__(self, buffer, sample_size=200):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        for experience in self.buffer.sample(self.sample_size):
            yield experience


def epsilon_greedy(state, env, net, epsilon=0.0):
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        # state = torch.tensor([state]).to(device)
        state = torch.tensor(state).to(device).unsqueeze(dim=0)
        q_values = net(state)
        _, action = torch.max(q_values, dim=1)
        action = int(action.item())
    return action


def create_environment(name, render_mode):
    env = gym.make(name, render_mode=render_mode)
    env = TimeLimit(env, max_episode_steps=400)
    env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda x: x % 1_000 == 0)
    env = RecordEpisodeStatistics(env)
    return env


class DeepQLearning(LightningModule):
    def __init__(self, env_name, render_mode=None, policy=epsilon_greedy, capacity=100_000, batch_size=254,
                 lr=1e-3, hidden_size=128, gamma=0.99, loss_fn=F.smooth_l1_loss, optim=AdamW,
                 eps_start=1.0, eps_end=0.15, eps_last_episode=100, samples_per_epoch=1_000,
                 sync_rate=10):
        super().__init__()
        self.env = create_environment(env_name, render_mode)

        # Create training and target network
        hidden_dim = hidden_size
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n
        self.q_net = DQN(hidden_dim, obs_dim, act_dim)
        self.target_q_net = copy.deepcopy(self.q_net)

        self.policy = policy
        self.buffer = ReplayBuffer(capacity=capacity)

        # Method from the LightningModule to save all provided arguments in self.hparams
        self.save_hyperparameters()

        # Populating the replay buffer with random actions by calling play_episode
        while len(self.buffer) < self.hparams.samples_per_epoch:
            print(f'{len(self.buffer)} samples in experience buffer. Filling...')
            self.play_episode(epsilon=self.hparams.eps_start)

    @torch.no_grad()
    def play_episode(self, policy=None, epsilon=0.0):
        state, _ = self.env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            if policy:
                action = policy(state, self.env, self.q_net, epsilon=epsilon)
            else:
                action = self.env.action_space.sample()
            next_state, reward, done, truncated, info = self.env.step(action)
            if truncated:
                done = truncated
            experience = (state, action, reward, done, next_state)
            self.buffer.append(experience)
            state = next_state

    # Forward
    def forward(self, x):
        return self.q_net(x)

    # Configure the optimizer
    def configure_optimizers(self):
        q_net_optimizer = self.hparams.optim(self.q_net.parameters(), lr=self.hparams.lr)
        return [q_net_optimizer]

    # Create dataloader
    def train_dataloader(self):
        dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            # num_workers=4
        )
        return dataloader

    # Training step
    def training_step(self, batch, batch_idx):
        states, actions, rewards, dones, next_states = batch
        actions = actions.unsqueeze(dim=1)
        rewards = rewards.unsqueeze(dim=1)
        dones = dones.unsqueeze(dim=1)

        # q(s,a)
        state_action_values = self.q_net(states).gather(dim=1, index=actions)

        next_action_values, _ = self.target_q_net(next_states).max(dim=1, keepdim=True)
        next_action_values[dones] = 0.0

        expected_state_action_values = rewards + self.hparams.gamma * next_action_values

        loss = self.hparams.loss_fn(state_action_values, expected_state_action_values)
        self.log('episode/Q-Error', loss)
        return loss

    # Training epoch end
    def training_epoch_end(self, training_step_outputs):

        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.current_epoch / self.hparams.eps_last_episode
        )

        self.play_episode(policy=self.policy, epsilon=epsilon)
        self.log('episode/Return', self.env.return_queue[-1].item())
        self.log('hp_metric', np.array(self.env.return_queue)[-100:].mean())

        if self.current_epoch % self.hparams.sync_rate == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())


def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    gamma = trial.suggest_float('gamma', 0.0, 1.0)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
    eps_end = trial.suggest_float('eps_end', 0.0, 0.3)
    sync_rate = trial.suggest_int('sync_rate', 10, 100, log=True)


    algo = DeepQLearning('LunarLander-v2',
                         render_mode='rgb_array_list',
                         lr=lr,
                         gamma=gamma,
                         hidden_size=hidden_size,
                         eps_end=eps_end,
                         sync_rate=sync_rate)

    callback = PyTorchLightningPruningCallback(trial, monitor='hp_metric')

    trainer = Trainer(
        gpus=num_gpus,
        max_epochs=400,
        callbacks=[callback]
    )

    hyperparameters = dict(lr=lr,
                           gamma=gamma,
                           hidden_size=hidden_size,
                           eps_end=eps_end,
                           sync_rate=sync_rate)


    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(algo)

    return trainer.callback_metrics['hp_metric'].item()


pruner = optuna.pruners.SuccessiveHalvingPruner()
study = optuna.create_study(direction='maximize', pruner=pruner)

study.optimize(objective, n_trials=5)

print(study.best_params)


# Train the policy
algo = DeepQLearning('LunarLander-v2', render_mode='rgb_array_list', **study.best_params)

trainer = Trainer(
    gpus=num_gpus,
    max_epochs=1_000,
    callbacks=[EarlyStopping(monitor='episode/Return', mode='max', patience=1_000)]
)

trainer.fit(algo)






