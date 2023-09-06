import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class DecisionTree:
    
    def __init__(self):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.val = 0
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        # Retrieve tests, labels
        y_categories = y.unique()
        print(y_categories)

        tests = X.columns.values

        test_tree = []
        while len(tests) > 0:
            print("These are the tests that remain", tests)
            
            # Generate a list for qualities
            test_qualities = []

            # Calculating the quality of each test
            for test in tests:
                X_test_df = X[test]
                categories = X[test].unique()
                total = len(X_test_df)
                
                # Want weights
                entropies = []
                sizes = []

                # Calculate entropy, and add these to a list
                # Should probably be a struct of some sort, but I am not bothering with it as of now.
                for category in categories:
                    category_df = X_test_df.loc[X_test_df == category]
                    
                    sizes.append(len(category_df))

                    index_entries = category_df.index.values.tolist()
                    
                    # Retrieve the output values
                    y_rows = y.iloc[index_entries]
                    
                    # Retireving the output matching either positive or negatative.
                    pos = len(y_rows.loc[y_rows == y_categories[0]])
                    neg = len(y_rows.loc[y_rows == y_categories[1]])
                    
                    entropy = calculate_entropy(pos, neg)
                    entropies.append(entropy)
                    # Notice that the entropy helps us create leaves and what gets carried through
                    # The branches that gives great results should filter out


                test_quality = 0.0
                for i in range(len(entropies)):
                    test_quality = entropies[i]*sizes[i]/total + test_quality

                test_qualities.append(test_quality)

            opt_index = test_qualities.index(min(test_qualities))
            opt_quality = min(test_qualities)

            opt_test = tests[opt_index]
            tests = np.delete(tests, opt_index, 0)

            test_tree.append(opt_test)
            

        print(test_tree)



        #raise NotImplementedError()
    

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
        raise NotImplementedError()
    
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
        raise NotImplementedError()


def calculate_entropy(positive: int, negative: int) -> float:
    total = positive + negative

    # Avoiding log2(0)
    if (positive > 0):
        pos_entropy = -(positive/total)*np.log2(positive/total)
    else:
        pos_entropy = 0
    
    # Avoiding log2(0)
    if (negative > 0):
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



