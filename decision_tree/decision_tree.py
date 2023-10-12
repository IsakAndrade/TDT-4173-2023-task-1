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
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature  # Index of feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.value = value  # Prediction value for leaf nodes
        self.left = left  # Left subtree
        self.right = right  # Right subtree
    

class DecisionTree:
    
    def __init__(self, max_depth:int = None):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        self.tree = self._grow_tree(X, y, depth=0)

        # # Retrieve tests, labels
        # y_categories = y.unique()
        # tests = X.columns.values
        # test_tree = []
        # tree = []
        # q_tree = [[]]
        # data_split = [X]
        # rules_count = 0

        # while (len(tests) > 0) & (len((data_split)) > 0) & (len(tree)<8):        
        #     data = data_split.pop()
        #     # Generate a list for qualities
        #     test_qualities = []
        #     leaves = []
        #     for test in tests:
        #         X_test_df = data[test]
        #         categories = data[test].unique()
        #         total = len(X_test_df)
        #         # Want weights
        #         entropies = []
        #         test_types = []
        #         sizes = []
        #         for category in categories:
        #             category_df = X_test_df.loc[X_test_df == category]
        #             sizes.append(len(category_df))
        #             index_entries = category_df.index.values.tolist()
                    
        #             # Retrieve the output values
        #             y_rows = y.iloc[index_entries]
                    
        #             # Retireving the output matching either positive or negatative.
        #             pos = len(y_rows.loc[y_rows == y_categories[0]])
        #             neg = len(y_rows.loc[y_rows == y_categories[1]])
                    
        #             # Test results 
                                        
        #             entropy = calculate_entropy(pos, neg)
                    
        #             entropies.append(entropy)
                    
        #             if (pos>neg):
        #                 test_types.append(y_categories[0])
        #             else:
        #                 test_types.append(y_categories[1])

                
        #         leaves.append(leaf(test, categories, entropies, test_types))
                
        #         test_quality = 0.0
        #         for i in range(len(entropies)):
        #             test_quality = entropies[i]*sizes[i]/total + test_quality

                
        #         test_qualities.append(test_quality)
                
        #     opt_index = test_qualities.index(min(test_qualities))
            


        #     opt_test = tests[opt_index]
        #     tests = np.delete(tests, opt_index, 0)
            
        #     test_tree.append(opt_test)

        #     # Use test to remove unwnanted data entries...
        #     test_entry = leaves[opt_index]
        #     leaf_name = leaves[opt_index].leaf_name
        #     cats = leaves[opt_index].categories
        #     entropies = leaves[opt_index].entropies
        #     results = leaves[opt_index].results
        #     # Create a dataframe copy to toss everything into :/
        #     en_threshhold = 0.2
            
        #     X_test_df = data[leaf_name]

        #     root = q_tree.pop()

        #     for i, cat in enumerate(cats):
        #         category_df = X_test_df.loc[X_test_df == cat]
        #         index_entries = category_df.index.values.tolist()
                
        #         if entropies[i] < en_threshhold:
        #             data.drop(index = index_entries, inplace = True)
        #             tree.append((root + [(leaf_name, cat)], results[i]))
        #         else:
        #             q_tree.append(root + [(leaf_name, cat)])
        #             data_split.append(data.loc[index_entries])
        #     rules_count += 1
        # self.tree = tree

    def _grow_tree(self, X, y, depth):
        # Check termination conditions
        if depth == self.max_depth or len(set(y)) == 1:
            return TreeNode(value=self._most_common_label(y))

        # Find the best split
        feature, threshold = self._find_best_split(X, y)

        # Split the data
        X_left, y_left, X_right, y_right = self._split_data(X, y, feature, threshold)

        # Recursively grow subtrees
        left_subtree = self._grow_tree(X_left, y_left, depth + 1)
        right_subtree = self._grow_tree(X_right, y_right, depth + 1)

        # Create and return the current node
        return TreeNode(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)
    
    def _find_best_split(self, X: pd.DataFrame, y: pd.DataFrame):
        
        cats = y.unique()
        feats = X.columns.values
        
        for feat in feats:
            feat_cats = X[feat].unique()

        
        # Implement your own method to find the best split (e.g., based on information gain or Gini impurity)
        # This involves iterating over features and thresholds to find the optimal split point
        # You can also consider other impurity measures or splitting criteria
        pass


    

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        # TODO: Implement 
        
        min_X = X.index.values.tolist()[0]  
        max_X = X.index.values.tolist()[-1]
        y_pred = list(range(max_X + 1))
        
        
        

        for test in self.tree:
            queries= []
            for i in range(0, len(test[0])):
                query = "(" + "`" + test[0][i][0] + "`" + "==" + "\"" + test[0][i][1] + "\""+ ")"
                queries.append(query)
            
            query = ""
            
            if (len(queries)) > 1:
                for i in range(0, len(queries)-1, 2):
                    query = query + queries[i] + " and " + queries[i+1]
            else:
                query = queries[0]

            
            index = X.query(query).index.values.tolist()            
            print(index)
            for i in index:
               
                y_pred[i] = test[-1]
        
        y_arr = np.array(y_pred[min_X:])
        return y_arr

    
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
        # TODO: Implement
        return self.tree


def calculate_entropy(positive: int, negative: int) -> float:
    total = positive + negative

    # Avoiding log2(0)
    if (positive > 0) & (negative != 0):
        pos_entropy = -(positive/total)*np.log2(positive/total)
    else:
        pos_entropy = -1
    
    # Avoiding log2(0)
    if (positive != 0) & (negative > 0):
        neg_entropy = -(negative/total)*np.log2(negative/total)
    else:
        neg_entropy = -1
        
    entropy = pos_entropy + neg_entropy
    return entropy

# --- Some utility functions 
    
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




def entropy(counts):
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



