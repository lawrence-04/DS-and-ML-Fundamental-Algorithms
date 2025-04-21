# Decision Tree
Decision trees are an extremely widely used type of model for both regression and classification on tabular data.

There are many extensions to the decision tree, such as random forest and gradient boosting, but we will implement the original.

## Theory
For classification on a given tabular dataset:
1. Find the best splitting point. This is done by considering how a given split across a given feature will impact the separation of the label. This can be measured in various way. We will use the [Gini impurity](https://en.wikipedia.org/wiki/Gini_coefficient).
    * Impurity is computed for each side of the split, then an overall impurity is computed by weighting this score based on the fraction of the data the split is.
    * Our goal is to *minimize* impurity (maximise purity).
1. Split the dataset based on this best split.
1. Repeat recursively. I.e. find the best split for *all* child nodes.
1. Keep going until all samples belong to a single class, or the maximum depth is reached.
1. Assign all leaf nodes the majority class label from the training data.

For regression, the method is the same. However, rather than Gini impurity, you would use [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) (or similar) to find the best split. And the leaf nodes are assigned the mean of the training data, rather than the majority class.
