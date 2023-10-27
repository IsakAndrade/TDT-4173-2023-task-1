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
    def __init__(self, feature: str = None, threshold: float = None, left = None, right = None, depth = None, *, value: int = None):
        self.feature   = feature    # Index of feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.depth     = depth
        self.left      = left       # Left subtree
        self.right     = right      # Right subtree
        self.value     = value      # Prediction value for leaf nodes
        
    def is_leaf_node(self):
        return self.value is not None 
    

class DecisionTree:
    
    def __init__(self, min_samples_split: int = 2, max_depth: int = 100, bagging: bool = False, bag_frac = 1.0, n_trees: int = 1):
        self.min_samples_split  = min_samples_split
        self.max_depth          = max_depth
        self.bagging            = bagging
        self.bag_frac           = bag_frac
        self.rules              = [[] for i in range(n_trees)]
        self.n_trees            = n_trees
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """

        # Adding a bagging stage aka splitting data even further
        if self.bagging:
            for i in range(self.n_trees):
                X_sample = X.sample(frac=self.bag_frac)
                self._grow_tree(X = X_sample, y = y, rule_set= i)
        else:
            self._grow_tree(X = X, y = y, rule_set= 0)

    def _grow_tree(self, X: pd.DataFrame, y: pd.Series, depth:int = 0, rule = [], rule_set: int = 1):
        n_samples  = len(X.index)
        n_labels = len(y.unique())
        
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
                leaf_value = self._most_common_label(y)
                self.rules[rule_set].append([rule] + [leaf_value])
                
                return 0
        feat_idxs = X.columns.values.copy()
        
        np.random.shuffle(feat_idxs)

        # Modify best split to take a set of blocked split.
        best_feat = self._best_split(X.copy(), y.copy(), feat_idxs)
        
        best_threshs = X.copy()[best_feat].unique()
        
        # Add a threshhold so that bad rules don't get added :/
        
        
        for thr in best_threshs: 
            
            X_col = X[best_feat]
            new_idxs = X_col[X_col == thr].index.to_list()
            new_rule = rule + [(best_feat, thr)]
            
            # Check the stopping criteria
            y_new = y.copy().loc[new_idxs]
            X_new = X.copy().loc[new_idxs]
        
            
            X_new = X_new.drop(labels= best_feat, axis = 1)
            self._grow_tree(X = X_new, y = y_new, depth = depth+1, rule = new_rule, rule_set =  rule_set)
            


        return 0
    
    def _most_common_label(self, y:pd.Series):
        return y.value_counts().idxmax()

    def _best_split(self, X: pd.DataFrame, y: pd.Series, feat_idxs: list[int]):
        best_gain = 10000
        split_idx, split_thresh = None, None
        
        
        for feat_idx in feat_idxs:
            X_col = X[feat_idx]
            
            gain = self._total_entropy(y, X_col)
            
            
            if gain < best_gain:
                best_gain = gain
                split_idx = feat_idx  
        
        
        return split_idx
    
    def _total_entropy(self, y: pd.Series, X_col: pd.Series):
        total_ent = 0
        n = len(y)
        
        grouped  = X_col.groupby(X_col)
        
        for name, group in grouped:

            weight = len(group.index)/n
            counts = y[group.index].value_counts().values

            ent = self._entropy(counts)
            
            total_ent = total_ent + weight*ent


        return total_ent

    def _split(self, X_col: pd.Series, split_thr: int) -> tuple[list[int], list[int]]:
        
        left_idxs   = X_col[X_col <= split_thr].index.to_list()
        right_idxs  = X_col[X_col > split_thr].index.to_list()
       
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
    
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """

        return self.rules[0]
    
    def predict(self, X: pd.DataFrame):
        
        
        
        #return  np.array([self._traverse_tree(X_numeric.loc[idx], self.root) for idx in X_numeric.index.tolist()])
        
        if self.bagging:
            
            y_list = [None for _ in list(range(self.n_trees))]

            for i, rule_set in enumerate(self.rules):
                
                y_list[i] = self._predict_with_rules(X.copy(), rule_set)

            y_bagged = pd.concat(y_list, axis=1)
            
            # Added majority rule
            row_modes = y_bagged.mode(axis='columns')
            
            
            y = row_modes[0]
            
        else:
            rules = self.rules[0]
            y = self._predict_with_rules(X, rules)

        return y
    
    def _predict_with_rules(self, X: pd.DataFrame, rules)-> pd.Series:
        
        y = pd.Series(index=X.copy().index, dtype = str)
        
        for rule in rules:
                conditions_unformated = rule[0]
                
                result= rule[1]
                
                conditions = X.index >= 0

                for i, cond in enumerate(conditions_unformated):
                    feat = cond[0]
                    thr = cond[1]
                    
                    conditions = conditions & (X[feat] == thr)
                
                idxs = X[conditions].index.to_list()
                for idx in idxs:
                    y.loc[idx] = result
                    
        
        return y




    
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








