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
    def __init__(self, feature: str = None, threshold: float = None, left = None, right = None, *, value: str = None):
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
        
        
        self.features = X.columns.values
        self.numeric_features = list(range(len(X.columns.values)))

        X_numeric = X.copy()
        X_numeric.columns = self.numeric_features

        thresh_mappings = []
        for numeric_feature, feature in enumerate(self.features):
            thresh = X[feature].unique() 
            thresh_numeric = list(range(len(thresh)))
            thresh_mapping = [dict(zip(thresh, thresh_numeric)) for num in thresh_numeric][0]
            thresh_mappings.append(thresh_mapping)
            
            data = X[feature]
            X_numeric[numeric_feature] = pd.Series([thresh_mapping[val] for val in data])


        self.tree = self._grow_tree(X_numeric, y)

    def _grow_tree(self, X: pd.DataFrame, y: pd.Series, depth:int = 0) -> TreeNode:
        n_samples, n_feats = len(X.index), len(X.columns.values)
        n_labels = len(y.unique())

        # Check the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples <self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return TreeNode(value = leaf_value)
        
        feat_idxs = np.random.choice(n_feats, len(self.numeric_features), replace = False)

        best_feat, best_thresh,  = self._best_split(X, y, feat_idxs)
        print(best_feat)
        print(best_thresh)
        X_col = X[best_feat]

        left_idx, right_idx = self._split(X_col, best_thresh)

        left = self._grow_tree(X.loc[left_idx], y.loc[left_idx], depth+1)
        right = self._grow_tree(X.loc[right_idx], y.loc[right_idx], depth+1)

        return TreeNode(best_feat, best_thresh, left, right)
    
    def _most_common_label(self, y:pd.Series):
        return y.value_counts().idxmax()

    def _best_split(self, X: pd.DataFrame, y: pd.Series, feat_idxs: list[int]):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_col = X[feat_idx]
            thresholds = X_col.unique()
            print("Thresh is,", thresholds)
            for thr in thresholds:
                # Calculate 
                gain = self._information_gain(y, X_col, thr)
                print("The gain is,", gain)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh
    
    def _information_gain(self, y: pd.Series, X_col: pd.Series, thr):
        # Parent entropy
        counts = y.value_counts().values
        parent_entropy = self._entropy(counts)
        
        # Create children
        left_idxs, right_idxs = self._split(X_col, thr)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Calculate the wheighted gain
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y.loc[left_idxs]), self._entropy(y.loc[right_idxs])
        
        child_entropy =(n_l/n)*e_l + (r_l/n)*e_r
        
        # Calculate Information Gain
        information_gain  = parent_entropy - child_entropy

        return information_gain

    def _split(self, X_col: pd.Series, split_thr: int) -> tuple[list[int], list[int]]:
        
        left_idxs   = X_col[X_col <= split_thr].index
        right_idxs  = X_col[X_col > split_thr].index
        return left_idxs, right_idxs

    def _entropy(self, counts: np.array):
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
            return node.value

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








