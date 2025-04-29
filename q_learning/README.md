# Q-Learning
Q-learning is a reinforcement learning algorithm that find the best action from a given state of an environment.

## Theory
The data used to optimise this model is the environment state, the action taken, and the reward received.

The Q-value is the expected cumulative discounted future reward for taking an action from a given state, assuming the agent follows the optimal strategy. I.e. the function we want to approximate is:

$$Q(S_t, A_t) = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots |_{(S_t, A_t)}$$

The update formula is given as follows:

$$Q^{new}(S_t, A_t) \larr Q(S_t, A_t) + \alpha \cdot (R_{t+1} + \gamma \cdot \max_a Q(S_{t+1}, a) - Q(S_t, A_t))$$

Where:
* $t$: time step
* $S_t$: state at time $t$
* $A_T$: action at time $t$
* $R_t$: reward at time $t$
* $Q(S_t, A_t)$: q-value for state $S$ and action $A$ at time $t$.
* $\alpha$: learning rate
* $\gamma$: discount factor

We see that $(Q(S_t, A_t) - \gamma \cdot \max_a Q(S_{t+1}, a))$ is our approximation for $R_t$ (since $Q$ is the future cumulative discounted reward.) Hence, we update the Q-value towards a correction prediction for $R_t$.

The simplest way to implement this is using a Q-table. This is where all state-action pairs are given a Q-value, stored in a table. Note that this only works when the state-action space is relatively small. If the states or actions are continuous, they must be discretised first.

So the algorithm is as follows:
1. Initialise the Q-table. The simplest initialisation is 0 for all state-actions pairs.
1. Choose and action based on the Q-table. There are various methods to sample an action, each having a tradeoff between exploring new moves and exploiting good moves. The simplest method is epsilon-greedy, where you have a probability $\epsilon$ to choose a random move, otherwise the agent chooses the move with the highest Q-value.
1. Apply the environment and update the state based on the new environment.
1. Repeat until a termination condition is met (reach the goal, maximum number of steps etc)
