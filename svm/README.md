# SVM
Support Vector Machines are a classification model that attempts to find a hyperplane in some vector space that separates points into their classes.

Wiki [here](https://en.wikipedia.org/wiki/Support_vector_machine).

## Theory
The simplest SVM is linear with a hard boundary. This only works for linearly separable data. Therefore we introduce a soft-margin, that allows for misclassifications.

Given a dataset:

$$\set{(x_1,y_1),(x_2,y_2),...,(x_n,y_n)}, \space x_i \in \R^d, \space y_i \in \set{-1,+1}$$

We want to find a hyperplane that separates the classes with the maximum margin. A hyperplane is given by:

$$f(x) = w^Tx + b = 0$$
* $w \in \R^d$: normal vector (normal to the hyperplane)
* $b$: offset (along normal vector)

The sign of $f(x_i)$ flips depending on which side of the margin the point is on.

Since $f(x)$ can be scaled with $w$ and $b$, we choose to make the support vectors (the points closest to the margin) satisfy:

$$y_i(w^Tx_i + b) \le 1$$

The distance from a point to a hyperplane is:

$$\frac{|w^Tx_i + b|}{\|w\|}$$

Therefore:

$$\text{Margin width} = \frac{+1 - (-1)}{\|w\|} = \frac{2}{\|w\|}$$

Since we want to maximise this margin (minimise $\|w\|$), and squaring gives smoother gradients. We also want to bring points to the correct side based on their label, so we use the hinge loss. Therefore our objective function becomes:

$$\min_{w,b,\xi} {\frac{1}{2}\|{w}^2\| + C\sum_{i=1}^{n}\max(0, 1-y_i(w^T x_i + b))}$$

Where:
* $\xi_i \in \R_{\ge 0}$: slack variable for each point
* $C \gt 0$: regularization parameter (trade-off between margin and slack)

The first term is minimising the norm of the weight vector, i.e., maximize the margin between classes. The second term penalises violations of the margin (i.e., allow some mistakes).

## Additional Theory
In practice, the problem is often simplified by forming a Lagrangian. Then, the primal formula can be rewritten to a dual formula:

$$\max_\alpha{\sum_{i=1}^{n}{\alpha_i} - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}{\alpha_i \alpha_j y_i y_j K(x_i, x_j)}}$$

With the condition:

$$0 \le \alpha_i \le C \space \text{for all $i$}$$

$$\sum_{i=1}^{n}{\alpha_i y_i} = 0$$

Where:
* $\alpha$: Lagrange multipliers (the solution gives one per training point)
* $C$: regularization parameter (controls margin vs misclassification trade-off)
* $K(x_i, x_j)$: kernel function (measure of similarity between 2 input data points)

The RBF kernel is given by:

$$K(x_i,x_j) = \exp{(-\gamma\|x_i - x_j\|^2)}$$

This kernel is powerful since it emulates computing the distance vector in an infinite dimensional space.
The [kernel trick](https://en.wikipedia.org/wiki/Kernel_method) is a method for casting the data into a higher dimension to make finding a suitable hyperplane more reliable.
