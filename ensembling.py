from base_ensemble import *
from utils import *
import math
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
np.random.seed(42)
import time
import numpy as np
import os

class Node():
    def __init__(self, splitting_feature=None, threshold=None, left=None, right=None, gain=None, prediction=None, is_leaf=None):
        self.splitting_feature = splitting_feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.prediction = prediction
        self.is_leaf = is_leaf
        
class Decision_Tree():
    def __init__(self, min_samples=2, max_depth=10):
        self.root = None
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.X_ = None
        self.y_ = None
        self.loss_function = None
        self.total_pos = None
        self.total_neg = None

    def leaf_prediction(self, y):
        if len(y) == 0:
            return 0 
        return 1 if np.sum(y == 1) > np.sum(y == -1) else -1


    def gini(self,left_pos, left_neg, right_pos, right_neg):
        left_total = left_pos + left_neg
        right_total = right_pos + right_neg
        total = left_total + right_total
        return 1 - (left_pos / left_total)**2 - (left_neg / left_total)**2 - (right_pos / right_total)**2 - (right_neg / right_total)**2
    
    def calc_entropy(self,pos, neg):
        total = pos + neg
        if total == 0:
            return 0
        p_pos = pos / total
        p_neg = neg / total
        return -p_pos * math.log2(p_pos + 1e-9) - p_neg * math.log2(p_neg + 1e-9)

    def entropy(self,left_pos, left_neg, right_pos, right_neg):
        left_total = left_pos + left_neg
        right_total = right_pos + right_neg
        total = left_total + right_total

        left_entropy = self.calc_entropy(left_pos, left_neg)
        right_entropy = self.calc_entropy(right_pos, right_neg)

        return (left_total / total) * left_entropy + (right_total / total) * right_entropy

    def information_gain(self, y_initial, y_left, y_right, method="entropy"):
        if method == "entropy":
            impurity_function = self.entropy
        elif method == "gini":
            impurity_function = self.gini_impurity
        else:
            raise ValueError("Method must be either 'entropy' or 'gini'")

        initial_impurity = impurity_function(y_initial)
        total_samples = len(y_initial)
        left_weight = len(y_left) / total_samples
        right_weight = len(y_right) / total_samples
        weighted_impurity = (left_weight * impurity_function(y_left)) + (right_weight * impurity_function(y_right))
        return initial_impurity - weighted_impurity

    def compute_gain_for_feature(self, X_copy, y_copy, j, gain_method):
        unique_vals = np.unique(X_copy[:, j])
        if len(unique_vals) < 2:
            return -1, None, None

        sorted_indices = np.argsort(X_copy[:, j])
        sorted_X = X_copy[sorted_indices, j]
        sorted_y = y_copy[sorted_indices]

        best_gain = 0.0
        threshold = None

        total_count = len(sorted_y)
        left_count = 0
        right_count = total_count

        left_pos = 0
        left_neg = 0
        right_pos = self.total_pos
        right_neg = self.total_neg

        impurity_function = self.entropy if gain_method == "entropy" else self.gini
        initial_impurity = impurity_function(left_pos, left_neg, right_pos, right_neg)

        for i in range(1, len(sorted_X)):
            label = sorted_y[i - 1]
            if label == 1:
                left_pos += 1
                right_pos -= 1
            else:
                left_neg += 1
                right_neg -= 1

            if sorted_y[i] != sorted_y[i - 1]:
                left_count = i
                right_count = total_count - left_count

                weighted_impurity = impurity_function(left_pos, left_neg, right_pos, right_neg)

                gain = initial_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    threshold = (sorted_X[i] + sorted_X[i - 1]) / 2

        return best_gain, j, threshold


    def best_split_feature(self, X, y, gain_method="entropy"):
        n_features = X.shape[1]
        best_gain = 0.0
        feature_idx = -1
        threshold = None

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(self.compute_gain_for_feature, X.copy(), y.copy(), j, gain_method)
                for j in range(n_features)
            ]

            for future in futures:
                gain, j_idx, th = future.result()
                if gain > best_gain:
                    best_gain = gain
                    feature_idx = j_idx
                    threshold = th

        return feature_idx, threshold, best_gain

    def train_helper(self, X, y, gain_method, depth=0):
        n_samples, n_features = X.shape
        prediction = self.leaf_prediction(y)

        if depth >= self.max_depth or n_samples <= self.min_samples:
            return Node(prediction=prediction, is_leaf=True)

        feature, threshold, best_gain = self.best_split_feature(X, y, gain_method)

        if best_gain == 0:
            return Node(prediction=prediction, is_leaf=True)

        left_indices = np.where(X[:, feature] <= threshold)[0]
        right_indices = np.where(X[:, feature] > threshold)[0]

        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        left_node = self.train_helper(X_left, y_left, gain_method, depth + 1)
        right_node = self.train_helper(X_right, y_right, gain_method, depth + 1)

        return Node(
            splitting_feature=feature,
            threshold=threshold,
            left=left_node,
            right=right_node,
            gain=best_gain,
            is_leaf=False,
            prediction=prediction,
        )

    def train(self, X, y, gain_method="entropy"):
        self.X_ = X
        self.y_ = y
        
        self.total_pos = np.sum(y == 1)
        self.total_neg = len(y) - self.total_pos
        self.loss_function = self.entropy
        if gain_method == "gini":
            self.loss_function == self.gini_impurity

        self.root = self.train_helper(X, y, gain_method)

    def inference_helper(self, x, node):
        if node.is_leaf:
            return node.prediction

        if x[node.splitting_feature] <= node.threshold:
            return self.inference_helper(x, node.left)

        return self.inference_helper(x, node.right)

    def inference(self, X):
        return [self.inference_helper(x, self.root) for x in X]

class RandomForestClassifier(BaseEnsembler):
    def __init__(self, num_trees=10, max_depth=50, min_samples=5, pca_components=58):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.pca = PCA(n_components=pca_components) 
        self.trees = []
        
        # print(num_trees)
        # print(max_depth)
        # print(min_samples)

    def _train_single_tree(self, X, y):
        tree = Decision_Tree(min_samples=self.min_samples, max_depth=self.max_depth)
        tree.train(X, y) 
        return tree

    def _train_tree_wrapper(self, sample):
        X_sample, y_sample = sample
        return self._train_single_tree(X_sample, y_sample)

    def fit(self, X, y):
        start_time = time.time() 

        # print("Starting Random Forest training...")
        n_samples, n_features = X.shape        
        X_pca = self.pca.fit_transform(X) 
        
        bootstrap_samples = []
        for _ in range(self.num_trees):
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X_pca[bootstrap_indices]  
            y_bootstrap = y[bootstrap_indices]
            bootstrap_samples.append((X_bootstrap, y_bootstrap))

        self.trees = []
        i = 1
        for X_bootstrap, y_bootstrap in bootstrap_samples:
            tree = self._train_tree_wrapper((X_bootstrap, y_bootstrap))
            # print(f"Tree {i} trained!")
            i+=1
            self.trees.append(tree)

        end_time = time.time() 

    def predict(self, X):
        start_time = time.time()  

        X_pca = self.pca.transform(X)  
        # print(f"Test data transformed using PCA.")

        all_predictions = []
        for tree in self.trees:
            tree_predictions = tree.inference(X_pca) 
            all_predictions.append(tree_predictions)

        all_predictions_array = np.array(all_predictions)
        final_predictions = np.where(all_predictions_array.sum(axis=0) > 0, 1, -1)
        end_time = time.time()  # End timing
        # print(f"Prediction completed in {end_time - start_time:.2f} seconds.")

        return np.array(final_predictions)
    

class AdaBoostClassifier(BaseEnsembler):

    def __init__(self, num_trees = 200):

        super().__init__(num_trees)
        self.learning_rate = 0.5
        self.max_depth = 1
        self.weak_classifiers = []  
        self.amount_of_say = []  
        self.n_samples = None
        self.sample_weights = None
        self.pca = PCA(n_components=114)

    def fit(self, X, y):
        '''
        Function signature SHOULD NOT BE CHANGED. Please maintain the output format and output sizes
        Args:
            X : input data. Shape : (no. of examples , flattened dimension)
            y : output data. Shape : (no, of examples, )

        Ouput:

            None
        '''
        n_samples = X.shape[0]
        X_pca = self.pca.fit_transform(X)  
        # print(X_pca.shape)
        self.sample_weights = np.ones(n_samples) / n_samples

        for i in range(self.num_trees):
            indices = np.random.choice(n_samples, size=n_samples, replace=True, p=self.sample_weights)
            X_sampled, y_sampled = X_pca[indices], y[indices]

            estimator = Decision_Tree(min_samples=2, max_depth=self.max_depth)
            estimator.train(X_sampled, y_sampled, gain_method="entropy")
            # print(f"Tree {i} trained!")
            predictions = estimator.inference(X_pca)

            incorrect = predictions != y
            error = np.dot(self.sample_weights, incorrect)

            if error == 0:
                break

            say = self.learning_rate * np.log((1 - error) / (error + 1e-10))
            self.weak_classifiers.append(estimator)
            self.amount_of_say.append(say)

            for i in range(n_samples):
                if incorrect[i]:
                    self.sample_weights[i] *= np.exp(say) 
                else:
                    self.sample_weights[i] /= np.exp(say) 

            self.sample_weights /= np.sum(self.sample_weights) 


    def predict(self, X):
        '''
        Function signature SHOULD NOT BE CHANGED. Please maintain the output format and output sizes

        Args:
            X : input data. Shape : (no. of examples , flattened dimension)
        Output:
            predictions : Shape : (no. of examples, )
        '''
        n_examples = X.shape[0]
        X_pca = self.pca.transform(X)
        # print(X_pca.shape)
        total_predictions = np.zeros(n_examples)
        
        for says, weak_classifier in zip(self.amount_of_say, self.weak_classifiers):
            weak_predictions = weak_classifier.inference(X_pca)
            weak_predictions = np.array(weak_predictions, dtype=np.float64)
            
            total_predictions += says * weak_predictions

        predictions = np.sign(total_predictions)
        predictions = np.where(predictions == 1, 1, -1)
        
        return predictions