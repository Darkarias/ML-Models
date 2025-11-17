# PS4: Exploratory Data Analysis and Batch Gradient Descent
Authors: Jacob Mongold and Victor Lockwood
Course: IDAI 610 – Fundamentals of Artificial Intelligence

## Overview
This problem set explores gradient descent methods and dimensionality reduction using the Titanic dataset. The notebook includes exploratory analysis, PCA, t-SNE, custom batch gradient descent, learning rate experiments, and synthetic data generation.

## Files
| File | Purpose |
|------|---------|
| gradient-notebook.ipynb | Full solution for all problem set questions |
| data/Titanic-Dataset.csv | Dataset used for analysis and modeling |
| synthetic_titanic.csv | Synthetic dataset used for analysis and modeling |
| README.md | Project description and usage information |
| requirements.txt | Python dependencies for running the notebook |

## Contents

### Part 1: Exploratory Data Analysis
- Load and inspect Titanic data
- Summaries, distributions, missing values
- Feature relationships and initial observations

### Problem 2: Visualizations
- Histograms for age and fare
- Bar plots for categorical variables
- Boxplots and pairplots
- Spearman correlation heatmap

### Problem 3: PCA and t-SNE
- Standardization with StandardScaler
- Explained variance ratios
- Cumulative variance plot
- 2D PCA projection
- 2D t-SNE projection
- Interpretation of structure in survival patterns

### Problem 4: Batch Gradient Descent
- Linear model for predicting fare from age
- MSE function and gradient derivations
- Weight and bias update rules
- Iteration loop with tracked errors
- Experiments with multiple learning rates
- MSE vs iteration plots

### Problem 6: Synthetic Data Generation
- Generate synthetic samples using learned model parameters
- Combine real and synthetic data
- Compare distributions using histograms and boxplots
- Discussion on similarity between real and synthetic data

## How to Run
Install dependencies:
   ```
   pip install -r requirements.txt
   ```
Launch Jupyter:
   ```
   jupyter notebook
   ```
Open gradient-notebook.ipynb and run all cells.

## Requirements
See requirements.txt for all dependencies. Core libraries include NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn.

## Notes
- All plots and outputs generate inside the notebook.
- The code is self-contained and requires no external modules.