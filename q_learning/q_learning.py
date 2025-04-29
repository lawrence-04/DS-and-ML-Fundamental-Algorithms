import numpy as np


class QLearning:
    def __init__(
        self,
        env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        n_bins: int = 50,
        seed: int = 42,
    ):
        """
         Q-learning parameters for CartPole

        Args:
            env: Gymnasium environment
            learning_rate: Alpha - learning rate
            discount_factor: Gamma - discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Rate at which epsilon decays after each episode
            epsilon_min: Minimum value of epsilon
            n_bins: Number of bins to discretize each state dimension
            state_bounds: Bounds for state discretization [(min1, max1), (min2, max2), ...]
        """
        np.random.seed(seed)

        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.n_actions = env.action_space.n

        # Set up state discretization
        self.n_bins = n_bins
        self.state_bounds = [
            (-2.4, 2.4),  # Cart position
            (-3.0, 3.0),  # Cart velocity
            (-0.3, 0.3),  # Pole angle
            (-3.0, 3.0),  # Pole angular velocity
        ]
        self.n_discrete_states = n_bins ** len(self.state_bounds)

        # Initialize Q-table with zeros
        self.q_table = np.zeros((self.n_discrete_states, self.n_actions))

        self.bins = []
        for bound in self.state_bounds:
            self.bins.append(np.linspace(bound[0], bound[1], n_bins + 1)[1:-1])

    def discretize_state(self, state):
        # Use digitize to find which bin each state dimension falls into
        discretized = []
        for i, s in enumerate(state):
            bin_index = np.digitize(s, self.bins[i])
            discretized.append(bin_index)

        # Convert discretized state to a single index
        discrete_state = sum([x * (self.n_bins**i) for i, x in enumerate(discretized)])
        return int(discrete_state)

    def select_action(self, state):
        # Exploration: choose a random action
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # Exploitation: choose the best action based on Q-values
        return np.argmax(self.q_table[state])

    def update_q_table(
        self, state: np.ndarray, action: int, reward: float, next_state, done: bool
    ):
        best_next_action = np.argmax(self.q_table[next_state])

        if done:
            # If terminal state, there's no future reward
            target = reward
        else:
            target = (
                reward
                + self.discount_factor * self.q_table[next_state, best_next_action]
            )

        # Update Q-value for the current state-action pair
        self.q_table[state, action] += self.learning_rate * (
            target - self.q_table[state, action]
        )

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, n_episodes: int = 1000) -> tuple[list[float], list[int]]:
        rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            discrete_state = self.discretize_state(state)
            done = False
            truncated = False
            total_reward = 0
            steps = 0

            while not (done or truncated):
                action = self.select_action(discrete_state)

                next_state, reward, done, truncated, _ = self.env.step(action)
                next_discrete_state = self.discretize_state(next_state)

                self.update_q_table(
                    discrete_state, action, reward, next_discrete_state, done
                )

                discrete_state = next_discrete_state

                total_reward += reward
                steps += 1

            self.decay_epsilon()

            rewards.append(total_reward)
            episode_lengths.append(steps)

        return rewards, episode_lengths
