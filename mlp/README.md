# Multilayer Perceptron
The Multilayer Perceptron (MLP) is a foundational building block for all of deep learning. It is a series of weight matrices that, when optimised, map a set of inputs to desired targets.

Wiki [here](https://en.wikipedia.org/wiki/Multilayer_perceptron).

## Theory:
MLPs attempt to model non-linear complex relationships by combining series of linear transformations with non-linear activation functions.

The MLP is comprised of 3 components:
* Input layer: this is out input data, for example, image pixels, tabular data etc.
* Hidden layers: one or more layers between the input and output layers. This layers apply a weighted sum over the previous layers, adds a bias, and apply a non-linear activation function (e.g. ReLU).
* Output layer: Produces the final prediction.

 
Each layer performs the following transformation:

$$l^{(i+1)} = f(W^{(i)}l^{(i)} + b^{(i)})$$

Where, for layer $i$:
* $l^{(i)}$ is the layer (with $l^{(0)}$ as the input vector)
* $W^{(i)}$ is the matrix of weights
* $b^{(i)}$ is the bias vector
* $f$ is the activation function

Many activation functions exist, but we will use the simplest one; the rectified linear unit:

$$\text{ReLU}(x) = \max(0, x)$$

In order to optimise these weights, we need a way to evaluate the output with some loss function we want to minimise. This depends on the task. To keep this simple, we will consider predicting a single continuous target (regression). Hence, we can use mean-squared error (MSE):

$$\text{L} = \frac{1}{2} (\hat{y} - y)^2$$

Where $\hat{y}$ is the prediction and $y$ is the target. The factor of a half makes the gradient more simple.

Now that we can evaluate our predictions, we can bring our predictions closer to the target using an optimisation algorithm called [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)(GD). The idea be GD is to take the gradient of the loss function with respect to a given weight, and then "step" that weight in the direction of the gradient (i.e. towards a local minimum).

We can calculate the gradient of the loss w.r.t a given weight with the chain rule:

$$\frac{\partial L}{\partial W^{(n)}} = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial W^{(n)}}$$

From our equation for MSE, we have:

$$\frac{\partial L}{\partial \hat{y}} = (\hat{y} - y)$$

A general formula for the gradient of a given weight in an MLP is too verbose for this repository. Instead, we can derive the result for a simple 2 layer MLP:

$$\hat y = W^{(2)}\text{ReLU}(W^{(1)}x + b^{(1)}) + b^{(2)}$$

The gradients are then
$$\frac{\partial \hat y}{\partial W^{(2)}} = \text{ReLU}(W^{(1)}x + b^{(1)})$$
$$\frac{\partial \hat y}{\partial b^{(2)}} = 1$$
$$\frac{\partial \hat y}{\partial W^{(1)}} = W^{(2)} (\text{ReLU}^{\prime}(W^{(1)}x + b^{(1)}))^{T}\cdot x^{T}$$
$$\frac{\partial \hat y}{\partial b^{(1)}} = W^{(2)} (\text{ReLU}^{\prime}(W^{(1)}x + b^{(1)}))^{T}$$

Where $\text{ReLU}^\prime$ is the gradient of the $\text{ReLU}$ function, which is the step function.

Once you have the gradient, the corresponding parameter is updated as follow:

$$W \leftarrow W - \alpha \frac{\partial \hat{y}}{\partial W^{(n)}}$$

Where $\alpha$ is the step-size parameter, which stops the algorithm updating the weights too quickly, increasing stability.
