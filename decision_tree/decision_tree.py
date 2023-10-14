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
        if depth == self.max_depth or len(set(y)) == 1 or (len(X) == 0):
            return TreeNode(value=self._most_common_label(y))

        # Find the best split
        print(X)
        feature, threshold = self._find_best_split(X, y)

        # Split the data
        X_left, y_left, X_right, y_right = self._split_data(X, y, feature, threshold)

        # Recursively grow subtrees
        left_subtree = self._grow_tree(X_left, y_left, depth + 1)
        right_subtree = self._grow_tree(X_right, y_right, depth + 1)

        # Create and return the current node
        return TreeNode(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)
    
    def _find_best_split(self, X: pd.DataFrame, y: pd.DataFrame)-> tuple[str, float]:
        
        y_cats = y.unique()
        feats = X.columns.values
        
        feat_scores = list(range(len(feats)))
        feat_entropies = list(range(len(feats)))

        for i, feat in enumerate(feats):
            
            feat_cats = X[feat].unique()

            entropies = []
            for cat in feat_cats:
                
                cat_df = X[feat].loc[X[feat] == cat]
                
                cat_df_index = cat_df.index.values.tolist()
                
                # Retrieve the output values
                y_rows = y.loc[cat_df_index]
                
                # Retrieving the output matching either positive or negative.
                pos = len(y_rows.loc[y_rows == y_cats[0]])
                neg = len(y_rows.loc[y_rows == y_cats[1]])
                
                # Test results 
                                    
                entropy = calculate_entropy(pos, neg)
                
                entropies.append(entropy)
            
            # Best feature should be the lowest?
            feat_scores[i] = sum(entropies)
            # Add partitioning score...
            feat_entropies[i] = entropies 

        best_split_index = feat_scores.index(min(feat_scores))
        best_feat = feats[best_split_index]
        threshold = min(feat_entropies[best_split_index])


        return best_feat, threshold
    
    def _split_data(self, X: pd.DataFrame, y: pd.DataFrame, feature: str, threshold: float)-> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        # This function should not be looking for anything
        # Only calculates the options
        y_cats = y.unique()
        X_left  = pd.DataFrame()
        y_left  = pd.Series()
        X_right = pd.DataFrame()
        y_right = pd.Series()

        feat_cats = X[feature].unique()
        
        for cat in feat_cats:
            cat_df =  X[feature].loc[X[feature] == cat]
            cat_df_index = cat_df.index.values.tolist()
            
            # Retrieve the output values
            y_rows = y.loc[cat_df_index]
            
            # Retrieving the output matching either positive or negative.
            pos = len(y_rows.loc[y_rows == y_cats[0]])
            neg = len(y_rows.loc[y_rows == y_cats[1]])

            entropy = calculate_entropy(pos, neg)

            if entropy <= threshold:
                X_new_left = X.loc[cat_df_index].drop(feature, axis =1)
                X_left = pd.concat([X_left, X_new_left], axis = 0)
                y_left = pd.concat([y_left, y_rows], axis = 0)
            else:
                X_new_right = X.loc[cat_df_index].drop(feature, axis =1)
                X_right = pd.concat([X_right, X_new_right], axis = 0)
                y_right = pd.concat([y_right, y_rows], axis = 0)
                
        # Removes the features that are good enough.
        # 1. Split data based on the feature...
        # 2. Check to see if it the partitioning meets the feature
        # 3. Those that do not accomplish the threshhold gets removed to left tree

        return X_left, y_left, X_right, y_right
    
    def _most_common_label(self, y: pd.Series) -> str:
        
        counts = y.value_counts()
        
        return counts.idxmax() 
    

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
        pos_entropy = 0
    
    # Avoiding log2(0)
    if (positive != 0) & (negative > 0):
        neg_entropy = -(negative/total)*np.log2(negative/total)
    else:
        neg_entropy = 0
        
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



