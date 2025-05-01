# PPO
Proximal Policy Optimisation is an extremely popular reinforcement learning algorithm. It tries to optimise a policy through interactions with an environment.

## Theory
The policy is a function that maps an input state to a probability distribution over actions.

If we use a machine learning algorithm to model the policy (probability of a given action), we denote the model by $\pi_{\theta}(a_t | s_t)$, where $s_t$ is the state at time $t$, $a_t$ is an available action at time $t$, and $\theta$ is the parameters of the model.

Then, the objective function is:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left(r_t(\theta) \hat{A}_t, \,\text{clip}\left(r_t(\theta), 1 - \epsilon, 1 + \epsilon\right) \hat{A}_t\right)\right]$$


Note that we are maximising this function.

Where:
* $r_t(\theta) = \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$ (ratio of the new policy to the old policy)
* $\hat{A}_t$: the advantage estimate at $t$
* $\epsilon$: clip range (~0.2)

The Advantage is given by:

$$\hat{A_t} = Q(s_t, a_t) - V(s_t)$$

Where:
* $Q(s_t, a_t)$: Q-value (expected discounted cumulative reward) for action $a_t$ in state $s_t$ at time $t$
* $V(s_t)$: expected value (discounted cumulative reward) for state $s_t$ sampling actions based on the policy.

### Intuition
We see that the advantage is comparing the value of a given action relative to all actions, for a given state. That is, if an action is good compared to other actions, the advantage will be increased, even if the action creates a negative reward. This allows us to find the best move in a bad situation.

$r_t(\theta)$ will be greater than 1 if the new policy is more likely to choose action $a_t$ in state $s_t$, and will be less than 1 the new policy is *less* likely to choose $a_t$ in state $s_t$.

So if the policy wants to take a certain move ($r_t > 1$), if it's a good move ($A_t > 0$) the objective function goes up, but if it's a bad move ($A_t < 0$), the objective function goes down.

The clipping restricts the policy updates for good moves, but there is no limit for bad moves.

### In Practice
For the advantage, the true $Q$ and $V$ are unknown, so we need to estimate it. The most common way to do this is using [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).

However, the simplest method is temporal difference estimate:

$$\hat{A_t} = r_t + \gamma V(s_{t+1}) - V(s_t)$$

Since $Q(s_t, a_t) = r_t + \gamma V(s_{t+1})$, where $r_t$ is the reward at time $t$.

We use an actor-critic paradigm. The actor is the policy model, that dictates the probability of a given action. The critic model approximates the value of the state. We use 2 separate models to increase stability of training.

The value model therefore has the following loss function:

$$L^{\text{value}} = \frac{1}{N} \sum_{t} (V(s_t) - R_t)^2$$

### Algorithm
1. Run the current policy on the environment to collect actions, states, rewards, policy outputs and value estimate for each step over several episodes.
1. Once we have the desired amount of data, compute the advantage for each time step for each run using the temporal difference estimate ($V$ is the value model).
1. Train the value function, subject to the loss function above.
1. Train the policy model, subject to the objective function above.
1. We repeat this using the updated policy until a termination condition is met (max training steps, objective plateau)
