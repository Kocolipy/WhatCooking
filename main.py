# Import the required libraries

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, precision_score
from sklearn.svm import SVC
import numpy as np
import os
import pathlib
import json

import utils

cwd = pathlib.Path(os.getcwd())

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    # ('svd', TruncatedSVD()),
    ('clf', OneVsRestClassifier(SVC())),
])

for i in [100, 250]:
    parameters = {
            # 'tfidf__use_idf': [False],
            # 'tfidf__norm': [None],
            # 'tfidf__max_features': [None],
            # 'tfidf__binary': [False, True],
            'tfidf__max_df': [i],
            # 'tfidf__min_df': [2, 5, 10],
            # 'svd__n_components': [i],
            'clf__estimator__kernel': ['rbf'],
            # 'clf__estimator__degree': [2,4,5,6,7,8],
            # 'clf__estimator__tol': [0.001, 0.0001, 0.00001],
            # 'clf__estimator__class_weight': ['balanced'],
            'clf__estimator__C': [1, 2, 5],
            'clf__estimator__gamma': ["scale", 1, 0.75, 1.25],
            # 'clf__estimator__C': [2],
            # 'clf__estimator__gamma': [1.25],
            # "clf__estimator__coef0": [0.5]
        }

    if __name__ == '__main__':
        print("Loading data ...")
        train, train_label = utils.loadPreprocessed()

        # train = [utils.preprocess(t) for t in train]


        # train = [" ".join(t) for t in train]

        # Label Encoding
        lb = LabelEncoder()
        oh_train_label = lb.fit_transform(train_label)

        print("Performing grid search ...")
        grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=10)
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

        utils.saveModel({"model": grid_search.best_estimator_, "labelencoder": lb}, "maxDF{0}-SVMOvR".format(i))