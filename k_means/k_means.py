import numpy as np 
import pandas as pd
import random
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)
import matplotlib.pyplot as plt

class KMeans:
    
    def __init__(self, K:int = 2, max_iters: int = 100, plot_steps = False, n_rerolls = 100):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.n_rerolls = n_rerolls
        # List of sample indices for each cluster.
        self.clusters = [[] for _ in range (self.K)]

        # List of sample indices for each cluster.
        self.centroids = [[] for _ in range (self.K)]
        
        
    def fit(self, X: pd.DataFrame) -> None:
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
       

       
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

        best_reroll_score = 0
        # Initialize 
        self.X = X

        self.n_samples, self.n_features = X.shape
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace =False)
        
        centroids = np.array([self.X[idx, :] for idx in random_sample_idxs])

        while len(centroids) < self.K:
            distances = cross_euclidean_distance(centroids, X)
            prob_vec = distances.min(axis = 0)
            prob_vec = prob_vec**2/np.sum(prob_vec**2)
            #Note: zero proability that new centorid is allready a centroid
            idx = np.append(idx, np.random.choice(X.shape[0], size=1, p = prob_vec)) 
            centroids = X[idx,:]
        
        self.centroids = centroids

        for i in range(self.n_rerolls):
            
            # Find the two centroids that are closest and pick the centoid with the lowest average distance to other centroids
            centroid_dist = cross_euclidean_distance(self.centroids)
            cetorid_dist_inf = centroid_dist + np.diag(np.repeat(np.inf, centroid_dist.shape[0])) # Add inf to diag
            worst_pair = np.unravel_index((cetorid_dist_inf).argmin(), cetorid_dist_inf.shape) # Find indexes of worst pair
            worst_ind = worst_pair[0] if (np.mean(centroid_dist[worst_pair[0]])<np.mean(centroid_dist[worst_pair[1]])) else worst_pair[1]

            # Assign the old centroid to be the one closest to the poinst that are furthest away from the current centroids
            worst_sample = self._point_away(worst_ind)
            
            
            self.centroids[worst_ind,:] = self.X[worst_sample,:]
              
            # Optimize clusters
            for i in range(self.max_iters):
                # Assign samples to the closests centroids
                self.clusters = self._create_clusters(self.centroids)

                if self.plot_steps:
                    self.plot()

                centroids_old = self.centroids
                self.centroids = self._get_centroids(self.clusters)

                if self._is_converged(centroids_old, self.centroids):
                    print(i)
                    break
            
            # Classify these sample as the index of the clusters
            z = self._get_cluster_labels(self.clusters)
            score = euclidean_silhouette(X,z)
            
            if score > best_reroll_score:
                best_reroll_score = score
                best_reroll_centroids = self.centroids

        self.clusters = self._create_clusters(self.centroids)
        z = self._get_cluster_labels(self.clusters)
        
        return  z

    def _get_cluster_labels(self, clusters):
        # Each sample will get 
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels.astype(int)

    def _point_away(self, unwanted_centroid_idx):
        distances = []
        centroids = np.delete(self.centroids, unwanted_centroid_idx, axis = 0)

        for idx, sample in enumerate(self.X):
            
            avrage_dist = np.mean([euclidean_distance(sample, centroid) for centroid in centroids])
            
            distances.append(avrage_dist)
        return np.argmax(distances)
        
    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroid.abs
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        # Assign the mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
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
            ax.scatter(*point, marker = "x", color = "black", linewidth = 2)

        plt.show()


    
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
       

        return self.centroids
    
    
    
    
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
    
    distortion = 0.0
    clusters = np.unique(z)
    
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
  