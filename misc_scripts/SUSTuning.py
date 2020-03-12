# Import the required libraries

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import os
import pathlib
import torch

import utils

cwd = pathlib.Path(os.getcwd())

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    # ('svd', TruncatedSVD()),
    ('clf', SVC()),
])

parameters = {
    # 'tfidf__use_idf': [True],
    # 'tfidf__norm': [None],
    # 'tfidf__max_features': [None],
    # 'tfidf__binary': [False, True],
    # 'tfidf__max_df': [0.5],
    # 'tfidf__min_df': [0.0],
    # 'svd__n_components': [1500, 2000],
    'clf__kernel': ['rbf'],
    # 'clf__estimator__degree': [2,4,5,6,7,8],
    # 'clf__estimator__tol': [0.001, 0.0001, 0.00001],
    # 'clf__estimator__class_weight': ['balanced'],
    'clf__C': [2],
    'clf__gamma': ["scale"],
    # "clf__estimator__coef0": [0.5]
}
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    print("Loading data ...")
    train, train_label = utils.loadFoldTrainData(1)
    train_label = [1 if t == "southern_us" else 0 for t in train_label]
    train = [" ".join(t) for t in train]

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

    utils.saveModel({"preprocess": "", "model":grid_search.best_estimator_, "labelencoder":""}, "southern_us")