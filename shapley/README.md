# Shapley Values
Shapley Values can be used to measure the importance of features in a prediction or set of predictions. The wiki is [here](https://en.wikipedia.org/wiki/Shapley_value).


## Theory
### Game Theory
Shapley values were first conceived in the context of game theory. In this context:
* A game involves a set of players working together to achieve a reward
* The Shapley value distributes this reward between players fairly by considering their marginal contributions across all combinations of players.

The formula for a Shapley value is given by:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} \left[ v(S \cup \{i\}) - v(S) \right]$$

Where:
* $S$: subset of players
* $N$: full set of players
* $v(S)$: value (reward) of subset $S$.


What this formula means is that, for a given player, $i$, we take all subsets of players that doesn't include $i$. Then, for each subset, we compute the difference in the reward when we include $i$ vs not including $i$.

So if including $i$ increases the reward, the Shapley value increases.

There is also a normalisation factor. This factor takes into account the number of permutations of players before and after player $i$. This ensures the contribution we get is averaged over the possible player permutations.

### Machine Learning
In machine learning, the context is changed as follows:
* "Players" are the input features to our model.
* "Reward" is the model output.

So we run the model with every subset of features to compute the Shapley value (contribution) for each feature. This can be provided for a single data point for local importance, or averaged over a dataset for global importance.

### Removing Features
There are several ways to remove features, some of which are dependent on model architecture. The simplest way to "remove" a feature is to replace it with the mean of that feature in the training data.

### Approximating Subsets
As we get more features, the number of subsets grows exponentially. It becomes unfeasible to compute the Shapley value for every single feature combination. The simplest optimisation we can make is to randomly sample the subsets. However, more complex and model specific implementations are available.



