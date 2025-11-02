"""
Naive Bayes text classification from scratch for two datasets:
- Movie Reviews (binary sentiment)
- Twenty Newsgroups subset (three classes)

This module provides:
- A simple tokenizer
- A Multinomial Naive Bayes implementation with eventual Laplace smoothing
- Evaluation utilities producing accuracy, macro-precision, macro-recall, and confusion matrices

Conventions:
- Probabilities are computed in log2 space to avoid underflow.
- OOV (out of vocabulary) tokens at test time are ignored.
- Confusion matrix rows are true labels, columns are predicted labels.
"""

import os
import pandas as pd
import numpy as np
import re
from typing import List
from collections import Counter
from typing import Dict

# Base directory for data
BASE = r"C:\ml_models\UncertaintyandBayesNetwork\Data"

# Subfolders
REV_DIR = os.path.join(BASE, "dataset_1_review")
NEWS_DIR = os.path.join(BASE, "dataset_1_newsgroup")

# Full paths to each CSV
REV_TRAIN = os.path.join(REV_DIR, "reviews_polarity_train.csv")
REV_TEST  = os.path.join(REV_DIR, "reviews_polarity_test.csv")
NG_TRAIN  = os.path.join(NEWS_DIR, "newsgroup_train.csv")
NG_TEST   = os.path.join(NEWS_DIR, "newsgroup_test.csv")

# Load the CSVs
reviews_train = pd.read_csv(REV_TRAIN)
reviews_test  = pd.read_csv(REV_TEST)
news_train = pd.read_csv(NG_TRAIN)
news_test  = pd.read_csv(NG_TEST)

# Handle NaN text entries
for df in (reviews_train, reviews_test, news_train, news_test):
    if "Text" in df.columns:
        df["Text"] = df["Text"].fillna("")

print("Movie Reviews train shape:", reviews_train.shape)
print("Movie Reviews test shape :", reviews_test.shape)
print(reviews_train.head(3))

print("\nNewsgroup train shape:", news_train.shape)
print("Newsgroup test shape :", news_test.shape)
print(news_train.head(3))

# Making the tokenizer
def simple_tokenize(text: str) -> List[str]:
    """
    Convert raw text to a list of tokens.

    Processing steps
    - Lowercase the string
    - Keep letters, digits, and apostrophes
    - Replace other chars with a single space
    - Split on whitespace

    Parameters
    ----------
    text : str
        Raw input string.

    Returns
    -------
    list[str]
        Token list. Empty list if input is empty after filtering.

    Examples
    --------
    >>> simple_tokenize("Don't stop; WON'T stop!")
    ["don't", 'stop', "won't", 'stop']
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return text.strip().split()

# Doing self checks to ensure that this works
print("\n[Tokenizer checks]")
print(simple_tokenize("This is... GREAT!!!")) # ['this', 'is', 'great']
print(simple_tokenize("don't stop; won't stop")) # ["don't", 'stop', "won't", 'stop']
print(simple_tokenize("")) # []

# NB skeleton + fit()
class NaiveBayesText:
    def __init__(self, laplace: bool = False):
        """
        Initializes the Naive Bayes classifier and parameter storage.

        Parameters
        ----------
        laplace : bool, optional
            If True, Laplace smoothing (+1) is used when calculating likelihoods. 
            Defaults to False.
        """
        self.laplace = laplace # Flag to enable/disable Laplace smoothing during training
        self.classes = [] # Stores the unique class labels
        self.class_priors_log: Dict[str, float] = {} # Stores log(P(c)), the log of the prior probability for each class
        self.class_word_counts: Dict[str, Counter] = {} # Stores the word counts per class
        self.class_total_tokens: Dict[str, int] = {} # Stores the total number of word tokens (for non-unique words) for each class
        self.vocab = {} # Stores the set of all unique words found within the training data
        self.vocab_size = 0 # Stores the size of vocabulary - used as the smoothing constant
        self.feature_log_probs: Dict[str, Dict[str, float]] = {} # Stores log(P(w|c)), the log liklihoods for prediction

    def fit(self, texts, labels):
        """
        Trains the Naive Bayes model by calculating class priors and collecting 
        feature counts from the input texts and labels.

        Parameters
        ----------
        texts : list
            A list of text strings (documents) used for training.
        labels : list
            A list of corresponding class labels for each text.

        Returns
        -------
        self : NaiveBayesText
            The trained model instance.
        """
        # Classes and log priors
        self.classes = sorted(list(set(labels))) # Identifies all unique class labels and stores them in a list
        counts = Counter(labels) # Counts the total number of occurrences for each class label
        total = len(labels) # Getting the total number of training documents
        self.class_priors_log = {c: np.log2(counts[c] / total) for c in self.classes}
        # Above calculates and stores P(c) for each class c
        
        # Per class token counts
        self.class_word_counts = {c: Counter() for c in self.classes}
        for t, y in zip(texts, labels): # Iterating through each text and its corresponding label
            self.class_word_counts[y].update(simple_tokenize(t)) # Tokenizing the text and then updating the word counts for the specific class

        # Total tokens per class
        self.class_total_tokens = {
            c: sum(self.class_word_counts[c].values()) for c in self.classes
        }

        # Global vocabulary
        # Initializing a set to collect all unique words accross all classes
        vocab = set()
        for c in self.classes: # add all unique words from the current class to the global vocab set
            vocab.update(self.class_word_counts[c].keys())
        self.vocab = {token: i for i, token in enumerate(sorted(vocab))}
        self.vocab_size = len(self.vocab)

        # Placeholder for the likelihoods
        self.feature_log_probs = {c: {} for c in self.classes}
        return self
    
# Quick sanity check
movies_X = ["the movie was great great", "boring movie", "really great film"]
movies_Y = ["pos", "neg", "pos"]
nb = NaiveBayesText().fit(movies_X, movies_Y)
print("\n[NB sanity]")
print("Classes:", nb.classes)
print("Log priors:", nb.class_priors_log)
print("Vocab size:", nb.vocab_size)
print("Tokens in 'pos':", sum(nb.class_word_counts["pos"].values()))
print("Tokens in 'neg':", sum(nb.class_word_counts["neg"].values()))

def logsumexp2(log_values):
    """
    Computes log2(sum(2^x)) for a list of log-probabilities x.

    This is used for numerically stable normalization (summing up log-probabilities) 
    in the final step of Naive Bayes prediction.

    Parameters
    ----------
    log_values : list[float]
        A list of log2-probabilities.

    Returns
    -------
    float
        The log2 of the sum of the input probabilities.
    """
    if not log_values:
        return -np.inf # If a token never appears in class c we store here, so any count of that token would "kill" that class score
    m = max(log_values) # Converting to NumPy array to vectorize
    shifted = np.array(log_values, dtype=float) - m
    return m + np.log2(np.sum(np.power(2.0, shifted)))

def add_likelihoods_no_smoothing(self):
    """
    Computes and stores per-class token log-likelihoods P(w|c) without Laplace smoothing.

    If a token's count is zero in a class, its log-likelihood is set to -inf 
    (log2(0)) to zero out the posterior probability for that class.
    """
    for c in self.classes:
        denom = self.class_total_tokens[c] # Total tokens for class c
        for token in self.vocab:
            num = self.class_word_counts[c][token] # Count of token in class c
            # No smoothing: zerp -> -inf, else log2(num/demon)
            self.feature_log_probs[c][token] = (-np.inf if num == 0 else np.log2(num / denom))
            
def predict_prob_log(self, text: str) -> Dict[str, float]:
    """
    Calculates the posterior log-probability log2(P(c|text)) for each class.

    The score is calculated as: log(P(c)) + sum(count(w) * log(P(w|c))).
    Out-of-vocabulary (OOV) tokens are ignored in the calculation.

    Parameters
    ----------
    text : str
        The raw text document to classify.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping class labels to their normalized posterior log2-probabilities.
    """

    tokens = simple_tokenize(text)
    token_counts = Counter([t for t in tokens if t in self.vocab])

    raw = {}
    for c in self.classes:
        s = self.class_priors_log[c] # Start at log prior (base 2)
        for token, count in token_counts.items():
            s += count * self.feature_log_probs[c][token]
        raw[c] = s
    # Normalize with base-2 log-sum-exp so that in linear spaces sum to 1
    lse = logsumexp2(list(raw.values()))
    return {c: (raw[c] - lse) for c in raw}

def predict(self, text: str) -> str:
    """
    Classifies a text document by returning the class with the highest 
    posterior log-probability.

    Parameters
    ----------
    text : str
        The raw text document to classify.

    Returns
    -------
    str
        The predicted class label (the one with the maximum log-probability).
    """
    logps = self.predict_prob_log(text)
    return max(logps.items(), key=lambda kv:kv[1])[0]

NaiveBayesText.add_likelihoods_no_smoothing = add_likelihoods_no_smoothing
NaiveBayesText.predict_prob_log = predict_prob_log
NaiveBayesText.predict = predict

print("\n[log sanity]")
movies_nb = NaiveBayesText().fit(
    ["the movie was great great", "boring movie", "really great film"],
    ["pos", "neg", "pos"]
)

movies_nb.add_likelihoods_no_smoothing()
print("Predict 'great great movie' ->", movies_nb.predict("great great movie"))
print("Predict 'boring and dull' ->", movies_nb.predict("boring and dull"))

def evaluate(nb: NaiveBayesText, texts, labels):
    """
     Parameters
    ----------
    nb : NaiveBayesText
        A fitted NaiveBayesText instance (with learned priors and likelihoods).
    texts : list[str]
        List of text samples to classify.
    labels : list[str]
        True class labels corresponding to each text.

    Returns
    -------
    dict
        A dictionary containing:
        - "classes": List of class labels (order used for confusion matrix indexing)
        - "confusion_matrix": 2D numpy array where rows = true labels, cols = predicted
        - "accuracy": Overall classification accuracy (float)
        - "macro_precision": Mean precision across all classes (float)
        - "macro_recall": Mean recall across all classes (float)

    Notes
    -----
    - Uses macro-averaging for precision/recall to handle class imbalance.
    - Rows/columns in the confusion matrix correspond to the ordering of nb.classes.
    """
    
    classes = nb.classes
    index = {c: i for i, c in enumerate(classes)} # Mapping class label -> row/col index
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)

    correct = 0
    for t, y_true in zip(texts, labels):
        y_pred = nb.predict(t)
        conf_matrix[index[y_true], index[y_pred]] += 1
        correct += int(y_pred == y_true)
    
    # Overall accuracy
    accuracy = correct / len(labels) if labels else 0.0

    # Computing per class, then average - macro precsion and recall
    precision, recall = [], []
    for i, c in enumerate(classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fp) > 0 else 0.0
        precision.append(prec)
        recall.append(rec)

    return{
        "classes": classes,
        "confusion_matrix": conf_matrix,
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision)) if precision else 0.0,
        "macro_recall": float(np.mean(recall)) if recall else 0.0
    }

# Verifying that everything works as intended
print("\n[Evaluate sanity]")
movies_eval = evaluate(movies_nb,
                       ["the movie was great great", "boring movie", "really great film"],
                       ["pos", "neg", "pos"])

print("Class: ", movies_eval["classes"])
print("Confusion matrix (rows=true, cols = pred): \n", movies_eval['confusion_matrix'])
print(f"Accuracy: {movies_eval['accuracy']:.3f} MacroP: {movies_eval['macro_precision']:.3f} MacroR: {movies_eval['macro_recall']:.3f}")

# Pulling the train and test splits without laplace smoothing
Xr_train = reviews_train["Text"].tolist()
yr_train = reviews_train["Label"].tolist()
Xr_test = reviews_test["Text"].tolist()
yr_test = reviews_test["Label"]. tolist()

# Training the model without smoothing
nb_reviews = NaiveBayesText(laplace=False).fit(Xr_train, yr_train)

# Computing the log likelihoods P(w|c)
nb_reviews.add_likelihoods_no_smoothing()

# Evaluating on the test set
res_reviews = evaluate(nb_reviews, Xr_test, yr_test)

# Printing the results of above
print("\n[Movie Reviews --> No Smoothing]")
print("Classes (order):", res_reviews["classes"])
print("Confusion matrix (rows = true, cols = pred):\n", res_reviews["confusion_matrix"])
print(f"Accuracy: {res_reviews['accuracy']:.4f}")
print(f"Macro Precision: {res_reviews['macro_precision']:.4f}")
print(f"Macro Recall: {res_reviews['macro_recall']:.4f}")

def add_likelihoods(self):
    """
    Computes and stores per-class token log-likelihoods P(w|c) with optional 
    Laplace smoothing (+1).

    If self.laplace is True, the formula is:
    P(w|c) = (count(w,c) + 1) / (sum(count(w',c)) + |V|)
    
    If self.laplace is False, the formula is:
    P(w|c) = count(w,c) / sum(count(w',c))
    """
    for c in self.classes:
        # Denominator depends on whether Laplace is on or not
        denom = self.class_total_tokens[c] + (self.vocab_size if self.laplace else 0)
        for token in self.vocab:
            # Numerator gets +1 only when Laplace is on
            num = self.class_word_counts[c][token] + (1 if self.laplace else 0)
            if not self.laplace and num == 0:
                # Unseen token in class -> log prob = -inf
                self.feature_log_probs[c][token] = -np.inf
            else:
                self.feature_log_probs[c][token] = np.log2(num / denom)

NaiveBayesText.add_likelihoods = add_likelihoods

# Evaluate with the leplace on movie reviews
print("\n[Movie Reviews | Laplace +1]")
nb_reviews_laplace = NaiveBayesText(laplace=True).fit(Xr_train, yr_train) # Calculating and storing each unique word and what class it belongs to
nb_reviews_laplace.add_likelihoods() # Identifying the entire training vocab and its size
res_reviews_laplace = evaluate(nb_reviews_laplace, Xr_test, yr_test)
print("Confusion matrix (rows=True, cols=pred):\n", res_reviews_laplace["confusion_matrix"])
print(f"Accuracy: {res_reviews_laplace['accuracy']:.4f}  "
      f"MacroP: {res_reviews_laplace['macro_precision']:.4f}  "
      f"MacroR: {res_reviews_laplace['macro_recall']:.4f}")


# Evaluating the news groups: No smoothing vs Laplace
Xn_train = news_train["Text"].tolist()
yn_train = news_train["Label"].tolist()
Xn_test = news_test["Text"].tolist()
yn_test = news_test["Label"].tolist()

print("\n [Newsgroups | No smoothing]")
nb_news_no_smoothing = NaiveBayesText(laplace=False).fit(Xn_train, yn_train)
nb_news_no_smoothing.add_likelihoods() # Acts like there is no smoothing
res_news_no_smoothing = evaluate(nb_news_no_smoothing, Xn_test, yn_test)
print("Confusion matrix (rows=true, cols=pred):\n", res_news_no_smoothing["confusion_matrix"])
print(f"Accuracy: {res_news_no_smoothing['accuracy']:.4f}  "
      f"MacroP: {res_news_no_smoothing['macro_precision']:.4f}  "
      f"MacroR: {res_news_no_smoothing['macro_recall']:.4f}")

print("\n[Newsgroups | Laplace +1]")
nb_news_laplace = NaiveBayesText(laplace=True).fit(Xn_train, yn_train)
nb_news_laplace.add_likelihoods()
res_news_laplace = evaluate(nb_news_laplace, Xn_test, yn_test)
print("Confusion matrix (rows=true, cols=pred):\n", res_news_laplace["confusion_matrix"])
print(f"Accuracy: {res_news_laplace['accuracy']:.4f}  "
      f"MacroP: {res_news_laplace['macro_precision']:.4f}  "
      f"MacroR: {res_news_laplace['macro_recall']:.4f}")