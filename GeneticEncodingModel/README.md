# Genetic Algorithm for the 0-1 Knapsack Problem
**Author:** Jacob Mongold  
**Course:** IDAI-610 – Fundamentals of Artificial Intelligence

## Overview
This project uses a Genetic Algorithm (GA) to solve the 0-1 Knapsack Problem, maximizing value under a weight limit. It includes comparisons between selection methods, mutation/crossover effects, and population sizes.

## Files
| File | Purpose |
|------|----------|
| `genome_ps2.py` | Core GA implementation |
| `plot_selection_trials.py` | Creates fitness comparison plots |
| `population_experiments.py` | Tests different population sizes |
| `config_1.txt` / `config_2.txt` | Problem definitions |
| `ps2_report.pdf` | Final written report |

## Key Features
- Binary chromosome representation  
- Roulette and Tournament selection  
- Single-point crossover & bit-flip mutation  
- Configurable mutation rate and population size  
- Reproducible results (`np.random.seed(1470)`)  

## Run
```bash
python genome_ps2.py
python plot_selection_trials.py
python population_experiments.py
```

## Dependencies
Python 3.10+, NumPy, Pandas, Matplotlib, tempfile
Install with:
```bash
pip install numpy pandas matplotlib
```

## Importing Libraries
import numpy as np
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## References
Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). *Grey Wolf Optimizer.* Advances in Engineering Software, 69, 46–61. https://pmc.ncbi.nlm.nih.gov/articles/PMC9643465/

DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

tempfile: https://docs.python.org/3/library/tempfile.html - I didnt know that we were allowed to copy and past the code given within the problem set, so I figured out how to use this.