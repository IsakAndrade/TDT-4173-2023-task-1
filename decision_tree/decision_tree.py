import numpy as np 
import pandas as pd

# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)
class DecisionTree:
    
    def __init__(self, min_samples_split: int = 2, max_depth: int = 100, bagging: bool = False, bag_frac: float = 1.0, n_trees: int = 1):
        self.min_samples_split  = min_samples_split
        self.max_depth          = max_depth
        self.bagging            = bagging
        self.bag_frac           = bag_frac
        self.rules              = [[] for i in range(n_trees)]
        self.n_trees            = n_trees
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """

        # Create a set of rules if bagging is wanted.
        if self.bagging:
            for i in range(self.n_trees):
                X_sample = X.sample(frac=self.bag_frac)
                self._grow_tree(X = X_sample, y = y, rule_set= i)
        else:
            self._grow_tree(X = X, y = y, rule_set= 0)

    def _grow_tree(self, X: pd.DataFrame, y: pd.Series, depth:int = 0, rule: list[list[tuple[str, str]], str] = [], rule_set: int = 1) -> None:
        """
        This function recursivly creates a ruleset. Mainly not a tree due to my inexperience with tree construction.  
        """
        n_samples  = len(X.index)
        n_labels = len(y.unique())
        
        # If the tree is too deep, there is a pure leaf or no ability to spli the sample further return.
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
                leaf_value = self._most_common_label(y)
                self.rules[rule_set].append([rule] + [leaf_value])        
                return

        # Getting the features of the column values.
        feat_idxs = X.columns.values.copy()
        np.random.shuffle(feat_idxs)

        # Modify best split to take a set of blocked split.
        best_feat = self._best_split(X.copy(), y.copy(), feat_idxs)
        best_threshs = X.copy()[best_feat].unique()
        
        # Add a threshhold so that bad rules don't get added :/
        for thr in best_threshs: 
            # Extract the best features threshold.
            X_col = X[best_feat]
            new_idxs = X_col[X_col == thr].index.to_list()
            
            # Add the rule to the existing rule.
            new_rule = rule + [(best_feat, thr)]
            
            # Check the stopping criteria.
            y_new = y.copy().loc[new_idxs]
            X_new = X.copy().loc[new_idxs]
        
            # Remove the feature so it may not be selected for future iterations.
            X_new = X_new.drop(labels= best_feat, axis = 1)
            self._grow_tree(X = X_new, y = y_new, depth = depth+1, rule = new_rule, rule_set =  rule_set)

        return
    
    def _most_common_label(self, y:pd.Series) -> int:
        """
        Return the most occuring value in a series.
        """
        return y.value_counts().idxmax()

    def _best_split(self, X: pd.DataFrame, y: pd.Series, feat_idxs: list[int]):
        """
        Iterate through all features and thresholds. Return the best scoring one.
        """
        
        best_entropy = 10000
        split_idx, split_thresh = None, None
        
        for feat_idx in feat_idxs:
            X_col = X[feat_idx]
            
            entropy = self._total_entropy(y, X_col)
                
            if entropy < best_entropy:
                best_entropy = entropy
                split_idx = feat_idx  

        return split_idx
    
    def _total_entropy(self, y: pd.Series, X_col: pd.Series) -> float:
        """
        Calculate the entropy of a column.
        """

        total_ent = 0
        n = len(y)
        
        grouped  = X_col.groupby(X_col)
        
        for name, group in grouped:
            weight = len(group.index)/n
            counts = y[group.index].value_counts().values

            ent = self._entropy(counts)
            total_ent = total_ent + weight*ent

        return total_ent

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
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Use the ruleset to predict future values.
        """
        
        # Use multiple machine learning algorithm and a majority vote. 
        if self.bagging:
            
            # A list that holds predictions
            y_list = [None for _ in list(range(self.n_trees))]

            # Use a different ruleset for each prediction.
            for i, rule_set in enumerate(self.rules):
                y_list[i] = self._predict_with_rules(X.copy(), rule_set)

            # Creating a dataframe holding all values
            y_bagged = pd.concat(y_list, axis=1)
            
            # Added majority rule
            row_modes = y_bagged.mode(axis='columns')
            
            # Pick the highest values, and a random tie value
            y = row_modes[0]
            
        else:
            # Picking the first and only ruleset of a non-bagged value.
            rules = self.rules[0]
            y = self._predict_with_rules(X, rules)

        return y
    
    def _predict_with_rules(self, X: pd.DataFrame, rules:list[list[list[tuple[str, str]], str]])-> pd.Series:
        
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








