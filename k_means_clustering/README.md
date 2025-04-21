# K-Means Clustering
K-means clustering is an algorithm that attempts to group together similar points in a vector space. 

Wiki [here](https://en.wikipedia.org/wiki/K-means_clustering).

## Theory
The algorithm goes as follows:
1. Initialise $n$ centroids randomly. These will be the centers of each cluster.
1. Assign each point to its closest centroid. This creates $n$ groups.
1. For each group, find the centre of mass (mean of positions).
1. This centre of mass becomes the new centroid.
1. Repeat these steps until the new centroids move less than some tolerance, relative to the previous centroids.
