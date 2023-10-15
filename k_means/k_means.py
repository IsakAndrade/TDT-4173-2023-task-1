import numpy as np 
import pandas as pd
import random
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)
import matplotlib.pyplot as plt

class KMeans:
    
    def __init__(self, K:int, max_iters: int = 100, plot_steps = False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        
        # List of sample indices for each cluster.
        self.clusters = [[] for _ in range (self.K)]

        # List of sample indices for each cluster.
        self.centroids = [[] for _ in range (self.K)]
        
        
    def fit(self, X: pd.DataFrame):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        self.X = X
        
        self.n_samples, self.n_features = len(X.index), len(X.columns.values)

        # Initialize 
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace =False)

        self.centroids = [self.X[idx] for id in random_sample_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to the closests centroids
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            centroids_old = self.centroid
            self.get_centroids = self.get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break
        
        # Classify thes sample as the index of the clusters
        return self._get_cluster_labels(self.clusters)


        # labels = X.columns.values
        # dim = len(X.columns.values)
        # height =len(X)
        
        # # Initializing with reasonable values
        # min_arr = np.zeros(dim)
        # max_arr = np.zeros(dim)
       
        # for i in range(dim):
        #     max_arr[i] = np.max(X[labels[i]])
        #     min_arr[i] = np.min(X[labels[i]])
        
        # # Initializing centroid coordinates.
        # centroids = np.zeros(shape = (height, K*dim), dtype=float)
        # print(centroids)
        # # It does not need to match that precisely 
        # # However it would be nice if the numbers matched a bit more :)
        # for j in range(K*dim):
        #     centroid = min_arr[0][j%dim] + max_arr[0][j%dim]*random.random()
        #     centroids[:,j] = np.full((height,), centroid, dtype=float)
        
        # ITERATIONS = 2000

        # for i in range(ITERATIONS):
        #     """
        #     Now we move on to calculate each distance for each centroid.
        #     """
        #     # Generate a set of ones that get multiplied by the K-points
        #     """
        #     This is a matrix where each column reprecents a distance from a centroid.S
        #     numpy.array([
        #         [d1_1, d1_2, ..., d1_n],
        #         [d2_1, d2_2, ..., d2_n],
        #                 .
        #                 .
        #                 .
        #         [dm_1, dm_2, ..., dm_n]
        #     ])
        #     """
        #     distances = np.zeros((height, K), dtype=float)
        
        #     for k in range(K):
        #         distances[:,k] = euclidean_distance(X, centroids[:,k:dim+k])
        
        #     list_of_list = []  
        #     for i in range(K):
        #         list_of_list.append([])
            
        #     for row in range(height):
        #         index = np.where(distances[row, :] == np.min(distances[row, :]))[0][0]
        #         # Create an entry that makes sense
        #         x = []
        #         labels = X.columns.values.tolist()
        #         for label in labels:
        #             x.append(X.iloc[row][label])
        #         list_of_list[index].append(x)
            
        #     centroids_1 = []
        #     for i in range(K):
        #         arr = np.array(list_of_list[i])
        #         x = []
        #         if len(arr.shape)> 1:
        #             centroids_1.append(np.mean(arr[:,0]))
        #             centroids_1.append(np.mean(arr[:,1]))
        #         else:
        #             centroid = min_arr[0][0] + max_arr[0][0]*random.random()
        #             centroids_1.append(centroid)
        #             centroid = min_arr[0][1] + max_arr[0][1]*random.random()
        #             centroids_1.append(centroid)
                    


        #     for j in range(K*dim):
        #         centroid = centroids_1[j]
        #         centroids[:,j] = np.full((height,), centroid, dtype=float)
        # # Finish of by reshaping the whole shabang
        

        # self.centroids = centroids
    def _get_cluster_labels(self, clusters):
        # Each sample will get 
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

    
    
    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroid.abs
        cluster = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):

            centroid_idx = self._closest_centroid(sample, centroid)

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroid]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        # Assugn the mean value of clusters to centroids
        centroids = np.zeros(self.K, self.n_features)
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis = 0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # Distances between old and new centroids, for all centroids.
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt. subplots(figsize = (12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter (*point)

        for point in self.centroids:
            ax.sxatter(*point, marker = "x", color = "black", linewidth = 2)

        plt.show()


    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        # Should simply calculate distance and pick the one with the shortest distance :)

        # Add hyperparameter centroids
        K = 8
        
        dim = len(X.columns.values)
        height =len(X)

        distances = np.zeros((height, K), dtype= float)

        for k in range(K):
            distances[:,k] = euclidean_distance(X, self.centroids[:,dim*k:(dim+dim*k)])

        categorized = np.zeros(shape = (height), dtype=int)
        for row in range(height):
            index = np.where(distances[row, :] == np.min(distances[row, :]))[0][0]
            categorized[row] = index

        return categorized
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        dim = 2
        K = 8

        return self.centroids[0].reshape(K,dim)
    
    
    
    
# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    print(X.shape[0])
    print(z.shape[0])
    distortion = 0.0
    clusters = np.unique(z)
    print(clusters)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        #distortion += ((Xc - mu)**2).sum(axis=0)
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
  