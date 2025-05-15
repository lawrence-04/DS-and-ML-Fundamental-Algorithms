import copy

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical
from gymnasium import Env


class PPODataset(Dataset):
    def __init__(self, states_tensor, actions_tensor, rewards_tensor, terminated_tensor, advantages_tensor):
        self.states_tensor =states_tensor
        self.actions_tensor =actions_tensor
        self.rewards_tensor =rewards_tensor
        self.terminated_tensor = terminated_tensor
        self.advantages_tensor =advantages_tensor
    
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
    def __init__(self, gamma: float = 0.1, batch_size: int = 64):
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

        advantages.append(self.rewards[-1] - next_value_out)

        advantages = np.array(advantages)

        # normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages
    
    def _export_to_tensors(self):
        advantages = self._compute_advantage()
        
        states_tensor = torch.from_numpy(np.stack(self.states)).to(torch.float32)
        actions_tensor = torch.LongTensor(self.actions).unsqueeze(1)
        rewards_tensor = torch.from_numpy(np.stack(self.rewards)).unsqueeze(1).to(torch.float32)
        terminated_tensor = torch.Tensor(self.terminated).unsqueeze(1).to(torch.float32)
        advantages_tensor = torch.Tensor(advantages).unsqueeze(1).to(torch.float32)

        return states_tensor, actions_tensor, rewards_tensor, terminated_tensor, advantages_tensor
    
    def build_dataloader(self):
        ppo_dataset = PPODataset(*self._export_to_tensors())
        self.dataloader = DataLoader(ppo_dataset, batch_size=self.batch_size, shuffle=True)

class PPO:
    def __init__(
        self,
        env: Env,
        actor: nn.Module,
        critic: nn.Module,
        max_training_samples: int = 2_048,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        batch_size: int = 64,
        num_epochs: int = 10,
        num_training_cycles: int = 30,
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
        self.num_training_cycles=num_training_cycles

        self.ppo_data = PPOData(gamma=gamma, batch_size=batch_size)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    @torch.no_grad()
    def _simulate_policy(self):
        self.ppo_data.clear()

        terminated = True
        episode_lengths = []
        episode_length = 0
        while self.ppo_data.n_samples <= self.max_training_samples:
            if terminated:
                state, *_ = self.env.reset()
                if episode_length:
                    episode_lengths.append(episode_length)
                episode_length = 0
            episode_length += 1

            state_tensor = torch.from_numpy(state).unsqueeze(0)
            policy_out = self.actor(state_tensor)[0]
            value_out = self.critic(state_tensor)[0]

            action = Categorical(policy_out).sample().item()

            next_state, reward, terminated, *_ = self.env.step(action)

            self.ppo_data.update(state=state, terminated=terminated, action=action, reward=reward, value_out=value_out)

            state = next_state
        
        print(f"Average episode length: {np.mean(episode_lengths)}")
        self.ppo_data.build_dataloader()

    def _optimise_critic(self):
        for _ in range(self.num_epochs):
            for batch in self.ppo_data.dataloader:
                states, next_states, _, rewards, terminated, advantages = batch
                self.critic_optimizer.zero_grad()

                out = self.critic(states)
                target = rewards + self.gamma * self.critic(next_states) * (1 - terminated)

                loss = nn.functional.mse_loss(out, target)
                loss.backward()

                self.critic_optimizer.step()


    def _optimise_actor(self):
        for _ in range(self.num_epochs):
            for batch in self.ppo_data.dataloader:
                states, _, actions, _, _, advantages = batch

                self.actor_optimizer.zero_grad()

                dist = Categorical(logits=self.actor(states))
                log_probs = dist.log_prob(actions)

                with torch.no_grad():
                    old_dist = Categorical(logits=self.prev_actor(states))
                    old_log_probs = old_dist.log_prob(actions)

                # more numerically stable than a direct ratio
                ratios = torch.exp(log_probs - old_log_probs)

                loss = -torch.mean(torch.min(ratios * advantages, ratios.clip(min=1 - self.epsilon, max=1+self.epsilon) * advantages))
                loss.backward()

                self.actor_optimizer.step()
    

    def train(self):
        for i in tqdm(range(self.num_training_cycles)):
            actor_copy = copy.deepcopy(self.actor)

            self._simulate_policy()

            self._optimise_critic()

            self._optimise_actor()

            self.prev_actor = actor_copy


