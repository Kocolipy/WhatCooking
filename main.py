# Import the required libraries

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
import numpy as np
import os
import pathlib

import utils

"""
Primary function to train SVM 
"""

cwd = pathlib.Path(os.getcwd())

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svd', TruncatedSVD()),
    ('clf', OneVsRestClassifier(SVC())),
])

parameters = {
        # # Bag-of-words
        # 'tfidf__use_idf': [False],

        # # Vocab Fine-tuning
        # 'tfidf__max_df': [100, 250],
        # 'tfidf__min_df': [2, 10, 20],
        # 'tfidf__min_df': [10],
        'svd__n_components': [2000],

        # # Linear SVC
        # 'clf__estimator__C': [0.1, 0.5, 1, 2, 5, 10],
        # 'clf__estimator__loss': ["hinge", "squared_hinge"],
        # 'clf__estimator__max_iter': [100000],

        # # Sigmoid Kernel
        # 'clf__estimator__kernel': ['sigmoid'],
        # 'clf__estimator__C': [1, 2, 5, 10],
        # 'clf__estimator__gamma': ["scale", 1, 0.75, 1.25],
        # "clf__estimator__coef0": [0, 0.1, 0.5, 1.0]

        # RBF Kernel
        'clf__estimator__kernel': ['rbf'],
        'clf__estimator__C': [2],
        'clf__estimator__gamma': [1.25],

        # # Poly Kernel
        # 'clf__estimator__degree': [4],
        # 'clf__estimator__C': [1, 2, 5, 10],
        # 'clf__estimator__gamma': ["scale", 0.1, 1],
        # "clf__estimator__coef0": [0, 0.1, 0.5, 1.0]
    }

print("Loading data ...")
train, train_label = utils.loadPreprocessed()

# # Alternatively load original and preprocess
# train, train_label = utils.loadOriginalTrain()
# train = [utils.preprocess(t) for t in train]

# # Without preprocessing
# train, train_label = utils.loadOriginalTrain()
# train = [" ".join(t) for t in train]

# Label Encoding
lb = LabelEncoder()
oh_train_label = lb.fit_transform(train_label)

print("Performing grid search ...")
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=30, verbose=10)
grid_search.fit(train, oh_train_label)
print("Best score: %0.5f" % grid_search.best_score_)
print("Best param:", grid_search.best_params_)
print()
for param, score in zip(grid_search.cv_results_['params'], grid_search.cv_results_["mean_test_score"]):
    print("Accuracy", score)
    print("---------------------")
    for k, v in param.items():
        print(k, ":", v)
    print()

# Save the best model found
utils.saveModel({"model": grid_search.best_estimator_, "labelencoder": lb}, "svd2000_svmOvR")