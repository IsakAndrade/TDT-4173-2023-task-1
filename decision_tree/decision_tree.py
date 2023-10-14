import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


# Tree
# Holds only 2 values.
#   Tests: 
#   Quality:
#   Classification: 
#   Pass on: 
# Tests and a threshold that is set by the user of the tree
class TreeNode:
    def __init__(self, feature: str = None, threshold: float = None, left: TreeNode = None, right: TreeNode = None, *, value: str = None):
        self.feature   = feature    # Index of feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.value     = value      # Prediction value for leaf nodes
        self.left      = left       # Left subtree
        self.right     = right      # Right subtree
        
        def is_leaf_node(self):
            return self.value is not None 
    

class DecisionTree:
    
    def __init__(self, min_samples_split: int = 2, max_depth: int = 100, n_features: list[str] = None):
        self.min_samples_split  = min_samples_split
        self.max_depth          = max_depth
        self.n_features         = n_features
        self.root               = None 
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """

        # Transforming the data so that it becomes easier to work with.
        X_numeric = pd.DataFrame
        
        self.features = X.columns.values
        
        for numeric_feature, feature in enumerate(self.features):
            thresh = X[feature].unique() 
            thresh_numeric = list(range(len(thresh)))
            string_to_number_mapping = [dict(zip(thresh, number)) for number in thresh_numeric]

            X_numeric[numeric_feature] = X[feature].factorize()




        self.tree = self._grow_tree(X_numeric, y)

    def _grow_tree(self, X: pd.DataFrame, y: pd.Series) -> TreeNode:
        n_samples, n_feats = len(X.columns.values)
        

        # Check the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples <self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return TreeNode(value = leaf_value)
        
        feat_idxs = np.random.choice(n_feats, self.n_feats, replace = False)

        best_feat_idx, best_thresh,  = _self_best_split(X, y, feat_idxs)

        left_idx, right_idx = self._split(X[best_feat_idx], best_thresh)
        
        return TreeNode(best_feat_idx, best_thresh, left_idx, right_idx)

    def _best_split(self, X: np.DataFrame, y: np.Series, feat_idxs: list[int]):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_col = X[self.features[feat_idx]]
            thresholds = X_col.unique()

            for thr in thresholds:
                # Calculate 
                gain = _self_information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        return split_idx, split_threshold
    
    def _information_gain(y: pd.Series, X_col: pd.Series, thr):
        # Parent entropy
        counts = y.value_counts()
        parent_entropy = self._entropy(y)
        
        # Create children
        left_idxs, right_idxs = self._split(X_col, thr)

        if len(left_idxs) or len(right_idxs):
            return 0
        
        # Calculate the wheighted gain
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(n_l), self._entropy(n_r)
        
        child_entropy =(n_l/n)*e_l + (r_l/n)*e_r
        
        # Calculate Information Gain
        information_gain  = parent_entropy - child_entropy

        return information_gain

    def _split(self, X_col: pd.Series, split_thr) -> tuple[list[int], list[int]]:
        # This must be changed
        left_idxs   = X_col[X_col <= split_thr]
        right_idxs  = X_col[X_col > split_thr]
        return left_idxs, right_idxs

    def _entropy(counts: np.array):
        """
        Computes the entropy of a partitioning

        Args:
            counts (array<k>): a lenth k int array >= 0. For instance,
                an array [3, 4, 1] implies that you have a total of 8
                datapoints where 3 are in the first group, 4 in the second,
                and 1 one in the last. This will result in entropy > 0.
                In contrast, a perfect partitioning like [8, 0, 0] will
                result in a (minimal) entropy of 0.0
                
        Returns:
            A positive float scalar corresponding to the (log2) entropy
            of the partitioning.

        """
        assert (counts >= 0).all()
        probs = counts / counts.sum()
        probs = probs[probs > 0]  # Avoid log(0)
        return - np.sum(probs * np.log2(probs))
    
    def predict(self, X: pd.DataFrame):
        return np.array([self._traverse_tree(x) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value()

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)
        




    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()








