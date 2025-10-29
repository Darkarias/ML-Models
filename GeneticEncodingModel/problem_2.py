import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# data setup (same as config file)
# each tuple = (weight, value)
S = np.array([
    (12, 240), (7, 180), (11, 200), (8, 160), (9, 180),
    (6, 120), (5, 100), (3, 60), (2, 40), (1, 20)])

weights = S[:, 0]
values = S[:, 1]
W = 35  # knapsack capacity

# feature = [weight, value, value/weight ratio]
# label = 1 if item is in optimal greedy solution, else 0
ratio = values / weights
features = np.column_stack((weights, values, ratio))

# create labels using simple greedy heuristic (non-GA baseline)
sorted_idx = np.argsort(-ratio)  # sort by descending value/weight
current_weight = 0
labels = np.zeros(len(weights))
for i in sorted_idx:
    if current_weight + weights[i] <= W:
        labels[i] = 1
        current_weight += weights[i]


# train/test split and model
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"Decision Tree classification accuracy (predicting inclusion): {acc:.3f}")

# evaluate knapsack selection
# predict full set
y_pred_full = model.predict(features)

# compute metrics for predicted solution
total_value = np.sum(values[y_pred_full == 1])
total_weight = np.sum(weights[y_pred_full == 1])

print("\n--- scikit-learn Decision Tree Knapsack Result ---")
print(f"Total Value: {total_value}")
print(f"Total Weight: {total_weight} / Capacity {W}")
print(f"Items Selected: {np.where(y_pred_full == 1)[0].tolist()}")
print(f"Feasibility: {'Feasible' if total_weight <= W else 'Infeasible'}")