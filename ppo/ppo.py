import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class PPODataset(Dataset):
    def __init__(
        self,
        states_tensor,
        actions_tensor,
        rewards_tensor,
        terminated_tensor,
        advantages_tensor,
    ):
        self.states_tensor = states_tensor
        self.actions_tensor = actions_tensor
        self.rewards_tensor = rewards_tensor
        self.terminated_tensor = terminated_tensor
        self.advantages_tensor = advantages_tensor

    def __len__(self):
        return self.rewards_tensor.shape[0] - 1

    def __getitem__(self, index):
        state = self.states_tensor[index]
        next_state = self.states_tensor[index + 1]
        action = self.actions_tensor[index]
        reward = self.rewards_tensor[index]
        terminated = self.terminated_tensor[index]
        advantage = self.advantages_tensor[index]

        return state, next_state, action, reward, terminated, advantage


class PPOData:
    def __init__(self, gamma: float = 0.99, batch_size: int = 64):
        self.gamma = gamma
        self.batch_size = batch_size
        self.clear()

    def __getitem__(self, index):
        return (
            self.states[index],
            self.terminated[index],
            self.actions[index],
            self.rewards[index],
            self.value_outputs[index].item(),
        )

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.n_samples = 0
        self.states = []
        self.terminated = []
        self.actions = []
        self.rewards = []
        self.value_outputs = []
        self.dataloader = None

    def update(self, state, terminated, action, reward, value_out):
        # unless we're just starting an episode, we have a new training sample
        if not self.terminated or not self.terminated[-1]:
            self.n_samples += 1

        self.states.append(state)
        self.terminated.append(terminated)
        self.actions.append(action)
        self.rewards.append(reward)
        self.value_outputs.append(value_out)

    def _compute_advantage(self):
        advantages = []
        next_value_out = 0  # Default value if loop doesn't execute

        for i in range(len(self) - 1):
            _, terminated, _, reward, value_out = self[i]

            # if terminated, the next value is a different episode
            if not terminated:
                *_, next_value_out = self[i + 1]
            else:
                next_value_out = 0

            # TD estimate
            advantage = reward + self.gamma * next_value_out - value_out
            advantages.append(advantage)

        # For the last transition
        if len(self) > 0:
            advantages.append(self.rewards[-1] - self.value_outputs[-1].item())

        advantages = np.array(advantages)

        # normalize advantages
        if len(advantages) > 1:  # Only normalize if we have more than one sample
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def _export_to_tensors(self):
        advantages = self._compute_advantage()

        states_tensor = torch.from_numpy(np.stack(self.states)).to(torch.float32)
        actions_tensor = torch.LongTensor(self.actions).unsqueeze(1)
        rewards_tensor = (
            torch.from_numpy(np.array(self.rewards)).unsqueeze(1).to(torch.float32)
        )
        terminated_tensor = torch.Tensor(self.terminated).unsqueeze(1).to(torch.float32)
        advantages_tensor = torch.Tensor(advantages).unsqueeze(1).to(torch.float32)

        return (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            terminated_tensor,
            advantages_tensor,
        )

    def build_dataloader(self):
        ppo_dataset = PPODataset(*self._export_to_tensors())
        self.dataloader = DataLoader(
            ppo_dataset, batch_size=self.batch_size, shuffle=True
        )


class PPO:
    def __init__(
        self,
        env: gym.Env,
        actor: nn.Module,
        critic: nn.Module,
        max_training_samples: int = 4096,  # Increased for more complex environment
        gamma: float = 0.99,
        epsilon: float = 0.2,
        batch_size: int = 64,
        num_epochs: int = 10,
        num_training_cycles: int = 100,  # Increased for more complex environment
    ):
        self.env = env
        self.actor = actor
        self.prev_actor = copy.deepcopy(actor)
        self.critic = critic
        self.max_training_samples = max_training_samples
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_training_cycles = num_training_cycles

        self.ppo_data = PPOData(gamma=gamma, batch_size=batch_size)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        # For tracking training progress
        self.avg_episode_lengths = []
        self.avg_episode_rewards = []

    @torch.no_grad()
    def _simulate_policy(self):
        self.ppo_data.clear()

        terminated = True
        truncated = False
        episode_lengths = []
        episode_rewards = []

        current_episode_length = 0
        current_episode_reward = 0

        while self.ppo_data.n_samples <= self.max_training_samples:
            if terminated or truncated:
                if current_episode_length > 0:
                    episode_lengths.append(current_episode_length)
                    episode_rewards.append(current_episode_reward)

                state, _ = self.env.reset()
                current_episode_length = 0
                current_episode_reward = 0

            current_episode_length += 1

            state_tensor = torch.from_numpy(state).unsqueeze(0).float()
            policy_out = self.actor(state_tensor)[0]
            value_out = self.critic(state_tensor)[0]

            action = Categorical(policy_out).sample().item()

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            current_episode_reward += reward

            self.ppo_data.update(
                state=state,
                terminated=terminated,
                action=action,
                reward=reward,
                value_out=value_out,
            )

            state = next_state

        if episode_lengths:
            avg_len = np.mean(episode_lengths)
            avg_reward = np.mean(episode_rewards)
            self.avg_episode_lengths.append(avg_len)
            self.avg_episode_rewards.append(avg_reward)

        self.ppo_data.build_dataloader()

    def _optimise_critic(self):
        for _ in range(self.num_epochs):
            for batch in self.ppo_data.dataloader:
                states, next_states, _, rewards, terminated, _ = batch
                self.critic_optimizer.zero_grad()

                out = self.critic(states)
                target = rewards + self.gamma * self.critic(next_states) * (
                    1 - terminated
                )

                loss = nn.functional.mse_loss(out, target)
                loss.backward()

                self.critic_optimizer.step()

    def _optimise_actor(self):
        for _ in range(self.num_epochs):
            for batch in self.ppo_data.dataloader:
                states, _, actions, _, _, advantages = batch

                self.actor_optimizer.zero_grad()

                dist = Categorical(logits=self.actor(states))
                log_probs = dist.log_prob(actions.squeeze(-1))

                with torch.no_grad():
                    old_dist = Categorical(logits=self.prev_actor(states))
                    old_log_probs = old_dist.log_prob(actions.squeeze(-1))

                # more numerically stable than a direct ratio
                ratios = torch.exp(log_probs - old_log_probs)

                loss = -torch.mean(
                    torch.min(
                        ratios * advantages.squeeze(-1),
                        ratios.clip(min=1 - self.epsilon, max=1 + self.epsilon)
                        * advantages.squeeze(-1),
                    )
                )
                loss.backward()

                self.actor_optimizer.step()

    def train(self):
        for i in tqdm(range(self.num_training_cycles)):
            actor_copy = copy.deepcopy(self.actor)

            self._simulate_policy()

            self._optimise_critic()

            self._optimise_actor()

            self.prev_actor = actor_copy

            # Evaluate periodically
            if (i + 1) % 10 == 0:
                self.evaluate(num_episodes=5)

        return self.avg_episode_lengths, self.avg_episode_rewards

    @torch.no_grad()
    def evaluate(self, num_episodes=10, render=False):
        total_rewards = []
        episode_lengths = []

        env = self.env
        if render:
            env = gym.make("Acrobot-v1", render_mode="human")

        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0
            episode_length = 0

            while not (done or truncated):
                state_tensor = torch.from_numpy(state).unsqueeze(0).float()
                policy_out = self.actor(state_tensor)[0]
                action = torch.argmax(policy_out).item()  # Use the most likely action

                state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                episode_length += 1

            total_rewards.append(total_reward)
            episode_lengths.append(episode_length)

        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(episode_lengths)

        if render:
            env.close()

        return avg_reward, avg_length
