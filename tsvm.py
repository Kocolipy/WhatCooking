from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import scipy
import numpy as np
import json
import os
import pathlib
import utils
import ltsvm.twinsvm as TSVM

cwd = pathlib.Path(os.getcwd())

parameters = {
        # 'tfidf__max_features': [1000],
        'clf__kernel': ['RBF'],
        'clf__gamma': [0.1],
        # 'clf__gamma': [1],
        'clf__C1': [0.1],
        'clf__C2': [0.1],
    }

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', TSVM.OVO_TSVM()),
])

train, train_label = utils.loadTrainData()
train = [" ".join(t) for t in train]


# Label Encoding
lb = LabelEncoder()
oh_train_label = lb.fit_transform(train_label)

print("Performing grid search ...")
grid_search = GridSearchCV(pipeline, parameters, scoring="accuracy", cv=5, n_jobs=1, verbose=10)
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

    utils.saveModel(grid_search.best_estimator_, "best_tsvm_OvO")