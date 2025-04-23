# PCA
Principal Component Analysis is a dimensionality reduction technique. It attempts to take a set of correlated features and reduce them into a smaller set of uncorrelated features, called principle components.

## Theory
The principle components are the axes in the data that have the highest variance.

Method:
1. Normalise: Since variance depends on scale, we first normalise the features by subtracting the mean and dividing by the standard deviation.
1. Covariance matrix: We then compute the Covariance matrix. This tells us how correlated features are (and a features variance).
1. Eigenvectors: We extract the Eigenvectors and Eigenvalues from the covariance matrix. The Eigenvectors tell us which relative contributions of features produce the highest variance. The Eigenvalue tells us what this variance is.
1. We take the Eigenvectors with the top `n` largest Eigenvalues, where `n` is the number of principle components we want.
