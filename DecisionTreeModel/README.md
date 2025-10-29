# Problem Set 1: Decision Trees for Clinical Data

Author: Jacob Mongold
Course: IDAI 610 - MS in Artificial Intelligence
Date: September 2025

---

## Contents
This submission includes:
1. **Written Report**: `ps1_mongold.pdf`
   - Answers to all problem set questions (Q1–Q15).
   - Includes discussion of implementation, challenges, dataset analysis, and comparisons.

2. **Code**: `jacob_tree.py`
   - Custom decision tree implementation built from scratch using NumPy and Pandas.
   - Supports entropy and gini impurity criteria, recursive tree-building, stopping conditions, and baseline comparison.
   - Produces accuracy, precision, and recall metrics for training, development, and test sets.

---

## Requirements
To run the code, install Python 3.10+ with the following libraries:

```bash
pip install numpy pandas scikit-learn
```

---

## How to Run the Code
1. Place the provided CSV files (`wdbc_train.csv`, `wdbc_dev.csv`, `wdbc_test.csv`) in the same directory as `jacob_tree.py`.
   - These files are provided in the assignment package under `Final_data/`.

2. Run the Python script from the terminal:

```bash
python jacob_tree.py

jupyter==1.0.0
numpy==1.26.3
pandas==1.5.1
scikit-learn==1.1.3
graphviz==0.20.1
seaborn
```

3. The script will:
   - Train decision trees using both entropy and gini criteria with depths [3, 5, 10].
   - Print development and test accuracies for each configuration.
   - Compute precision and recall for malignant (M) as the positive class.
   - Output baseline classifier results for comparison.

---

## Expected Output
Example terminal output (values may vary slightly):

```
Criterion = entropy, Depth = 3, Dev Accuracy = 0.974
Criterion = entropy, Depth = 3, Test Accuracy = 0.947
Criterion = entropy, Depth = 3, Test Precision = 0.974, Test Recall = 0.881
Criterion = entropy, Depth = 5, Dev Accuracy = 0.965
Criterion = entropy, Depth = 5, Test Accuracy = 0.939
Criterion = entropy, Depth = 5, Test Precision = 0.949, Test Recall = 0.881
Criterion = entropy, Depth = 10, Dev Accuracy = 0.956
Criterion = entropy, Depth = 10, Test Accuracy = 0.921
Criterion = entropy, Depth = 10, Test Precision = 0.902, Test Recall = 0.881
Criterion = gini, Depth = 3, Dev Accuracy = 0.930
Criterion = gini, Depth = 3, Test Accuracy = 0.939
Criterion = gini, Depth = 3, Test Precision = 1.000, Test Recall = 0.833
Criterion = gini, Depth = 5, Dev Accuracy = 0.965
Criterion = gini, Depth = 5, Test Accuracy = 0.965
Criterion = gini, Depth = 5, Test Precision = 1.000, Test Recall = 0.905
Criterion = gini, Depth = 10, Dev Accuracy = 0.956
Criterion = gini, Depth = 10, Test Accuracy = 0.939
Criterion = gini, Depth = 10, Test Precision = 0.927, Test Recall = 0.905
Majority baseline results:
Accuracy = 0.632, Error = 0.368, Precision (M) = 0.000, Recall (M) = 0.000
```

---

## Notes
- The code is fully self-contained and does not require scikit-learn for tree building (only for metrics comparison).

---
