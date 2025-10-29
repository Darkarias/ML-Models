import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

def entropy(labels):  # compute entropy of label distribution
    values, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()  # probability of each variable in the count
    return -np.sum(probabilities * np.log2(probabilities))  # entropy formula

def gini(labels):  # compute Gini impurity.
    values, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities **2)

def information_gain(parent_labels, left_labels, right_labels, criterion = "entropy"):
    # figure out how much information is gained
    impurity = entropy if criterion == "entropy" else gini #chooses which impurity metric that is being used to split nodes
    parent_impurity = impurity(parent_labels)  # impurity before the split (the whole set)

    # weighted impurity after the split (children)
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)

    if n_left == 0 or n_right == 0: #if one side is empty, nothing was gained
        return 0

    child_impurity = (n_left / n) * impurity(left_labels) + (n_right / n) * impurity(right_labels)
    return parent_impurity - child_impurity  # difference = information gain

def split_dataset(data, feature_index, threshold):
    # split data into left and right subsets based on a feature and threshold
    left_mask = data[:, feature_index] < threshold
    right_mask = ~left_mask  # everything not in left goes right
    left_data = data[left_mask]
    right_data = data[right_mask]
    return left_data, right_data

class Leaf:
    def __init__(self, labels):
        #store what happens at a leaf: labels and majority class prediction
        values, counts = np.unique(labels, return_counts=True)
        probabilities = counts / counts.sum()
        self.classes = values
        self.probabilities = probabilities
        self.prediction = values[np.argmax(counts)] #majority label

class DecisionNode:
    def __init__(self, feature_index, threshold, left, right):
        #internal node in the tree with feature, threshold, and branches
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

def build_tree(data, depth = 0, max_depth = 5, min_samples = 2, criterion="entropy"):
    labels = data[:, -1]  # get the target label column

    # stopping conditions
    if len(np.unique(labels)) == 1: #checks if all labels in current dataset are identical
        return Leaf(labels) #if above is true, it stops splitting and returns a leaf as an object
    if depth >= max_depth or len(data) < min_samples: #hit depth limit or too few samples
        return Leaf(labels)  # stops the splitting

    # search for the best split
    best_gain = 0
    best_feature = None
    best_threshold = None
    best_left, best_right = None, None

    n_features = data.shape[1] - 1  # all columns except label
    for feature_index in range(n_features):
        thresholds = np.unique(data[:, feature_index]) # possible cut points for this feature
        for threshold in thresholds:  # each unique value is then tested for potential threshold to split the data
            left, right = split_dataset(data, feature_index, threshold)  # this is the actual split in the data
            if len(left) == 0 or len(right) == 0:
                continue
            gain = information_gain(labels, left[:, -1], right[:, -1], criterion)
            if gain > best_gain: #keeping track of best split so far
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold
                best_left, best_right = left, right
    
    

    # if there wasn't a good split
    if best_gain == 0: # if no split improved the information gain
        return Leaf(labels) # stop and make a leaf node

    # Recursively build the subtrees.
    left_subtree = build_tree(best_left, depth + 1, max_depth, min_samples, criterion)
    right_subtree = build_tree(best_right, depth + 1, max_depth, min_samples, criterion)
    
    return DecisionNode(best_feature, best_threshold, left_subtree, right_subtree)

def predict_one(sample, node): #predict label for a single sample by traversing the tree
    if isinstance(node, Leaf): #checks to validate if current node is a leaf or not
        return node.prediction #if leaf, then this returns the prediction stored within the leaf
    
    if sample[node.feature_index] < node.threshold: #if node != leaf, this compares the value of the sample at that feature to node.threshold
        return predict_one(sample, node.left) #if previous line is True, function is called again, but now moves to the left child
    else:
        return predict_one(sample, node.right) #handles the case where the condition returns False

def predict(data, tree_root): #takes the entire array and root node and applies predict_one function to every row within the dataset
    return np.array([predict_one(sample, tree_root) for sample in data])

class DecisionTree:
    def __init__(self, max_depth = None, min_samples = 2, criterion = "entropy"):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.criterion = criterion
        self.root = None

    def fit(self, X, y): #combining X and y into one array into the training label column
        data = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        self.root = build_tree(data, depth=0, max_depth=self.max_depth,
                       min_samples=self.min_samples,
                       criterion=self.criterion)
    
    def predict(self, X):
        return np.array([predict_one(sample, self.root) for sample in X])

# loading the data from the .csv files.
train = pd.read_csv("Final_data\wdbc_train.csv")
dev = pd.read_csv("Final_data\wdbc_dev.csv")
test = pd.read_csv("Final_data\wdbc_test.csv")

# features of all columns except diagnosis for train, dev, and test.
X_train = train.drop(columns = ["Diagnosis"]).values
y_train = train["Diagnosis"].map({"B":0, "M":1}).values

X_dev = dev.drop(columns = ["Diagnosis"]).values
y_dev = dev["Diagnosis"].map({"B":0, "M":1}).values

X_test = test.drop(columns =["Diagnosis"]).values
y_test = test["Diagnosis"].map({"B":0, "M":1}).values

# trying different hyperparameters
for criterion in ["entropy", "gini"]:
    for depth in [3, 5, 10]:
        tree = DecisionTree(max_depth = depth, criterion = criterion)
        tree.fit(X_train, y_train)

        prediction = tree.predict(X_dev) #dev accuracy
        accuracy = np.mean(prediction == y_dev)
        print(f"Criterion = {criterion}, Depth = {depth}, Dev Accuracy = {accuracy:.3f}")

        test_predict = tree.predict(X_test)
        test_accuracy = np.mean(test_predict == y_test)
        print(f"Criterion = {criterion}, Depth = {depth}, Test Accuracy = {test_accuracy:.3f}")

        # Compute precision and recall with malignant (1) as positive class
        test_precision = precision_score(y_test, test_predict, pos_label=1)
        test_recall = recall_score(y_test, test_predict, pos_label=1)

        print(f"Criterion = {criterion}, Depth = {depth}, "
              f"Test Precision = {test_precision:.3f}, Test Recall = {test_recall:.3f}")
        
# majority baseline always predicts the must frequent class in training
unique, counts = np.unique(y_train, return_counts=True)
majority_class = unique[np.argmax(counts)]

# create predictions for the test set by filling an array with the majority_class value
# I originally tried to iterate through arrays manually, but it gave me bugs
# np.full_like was a cleaner way to do it since it just builds the array in one line
baseline_preds = np.full_like(y_test, majority_class)

baseline_acc = np.mean(baseline_preds == y_test) # error is just 1 - accuracy
baseline_error = 1 - baseline_acc

# compute precision and recall for malignant (1) as the positive class
# tp = true positives, fp = false positives, fn = false negatives
tp = np.sum((baseline_preds == 1) & (y_test == 1))
fp = np.sum((baseline_preds == 1) & (y_test == 0))
fn = np.sum((baseline_preds == 0) & (y_test == 1))

# precision is tp / (tp+fp), recall is tp / (tp+fn)
# added checks to avoid divide by zero errors
baseline_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
baseline_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

print(f"Majority baseline results: ")
print(f"Accuracy = {baseline_acc:.3f}, Error = {baseline_error:.3f}, Precision (M) = {baseline_precision:.3f}, Recall (M) = {baseline_recall:.3f} ")

# Referenced documentation:

# recall_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
# precision_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
# random_state: https://scikit-learn.org/stable/glossary.html#term-random_state

# Numpy: https://numpy.org/doc/stable/user/basics.html
# array-manipulation: https://numpy.org/doc/stable/reference/routines.array-manipulation.html
# full_like: https://numpy.org/doc/stable/reference/generated/numpy.full_like.html#numpy-full-like