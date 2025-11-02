# Naive Bayes Text Classifier (from scratch)
**Author:** Jacob Mongold 
**Course:** IDAI-610 – Fundamentals of Artificial Intelligence

## Overview
A from-scratch Multinomial Naive Bayes for two text datasets:
- Movie Reviews (binary sentiment)
- Twenty Newsgroups subset (3 topics)

Implements tokenization, priors, likelihoods, optional Laplace smoothing, and an evaluation utility that returns accuracy, macro precision, macro recall, and a confusion matrix.

## Files
| File | Purpose |
|------|---------|
| `bayes.py` | Main module with tokenizer, Naive Bayes, and evaluation utilities |
| `README.md` | Project documentation |
| `requirements.txt` | Minimal dependencies |

## Key Features
- Multinomial Naive Bayes in log space
- Optional Laplace smoothing (+1)
- OOV tokens ignored at test time
- Macro-averaged precision and recall
- Confusion matrix with rows = true, cols = predicted
- Small, readable codebase

## Run
Quick sanity check without datasets:
```bash
python bayes.py
```

## Dependencies
Python 3.10+ with NumPy and pandas.
Install with:
```bash
pip install -r requirements.txt

or

pip install numpy pandas
```

## Notes on documentation
This is the first project where I’ve used docstrings throughout the code. I’m adopting them because they’re the industry standard and make the code easier to read and maintain for other developers and reviewers. In future projects, I plan to rely primarily on docstrings and reserve inline comments only for sections that truly need clarification.

## References
- Movie Reviews dataset: https://www.cs.cornell.edu/people/pabo/movie-review-data/
- Twenty Newsgroups subset: https://archive.ics.uci.edu/dataset/113/twenty+newsgroups
- Python docstrings standard examples: https://www.programiz.com/python-programming/docstrings, https://www.geeksforgeeks.org/python/python-docstrings/, 
    https://www.datacamp.com/tutorial/docstrings-python