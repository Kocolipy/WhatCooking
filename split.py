# Import the required libraries

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from scipy.sparse import vstack
import os
import pathlib
import utils

cwd = pathlib.Path(os.getcwd())

pipeline = Pipeline([
    # ('tfidf', TfidfVectorizer()),
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
        'clf__C': [0.5, 1,  2, 5, 10, 25, 100],
        'clf__gamma': ["scale", 1.1, 0.9, 1, 0.1, 1.5, 2, 0.5, 5],
        # "clf__estimator__coef0": [0.5]
    }

tree = [['XTTBT', ['vietnamese', 'korean', 'japanese', 'thai', 'filipino', 'chinese'], ['russian', 'indian', 'italian', 'french', 'spanish', 'mexican', 'southern_us', 'moroccan', 'british', 'greek', 'cajun_creole', 'jamaican', 'brazilian', 'irish']], ['EMRXQ', ['vietnamese', 'thai'], ['korean', 'japanese', 'filipino', 'chinese']], ['UBDET', ['vietnamese'], ['thai']], ['HUBPG', ['korean', 'japanese', 'chinese'], ['filipino']], ['KJMWA', ['korean', 'chinese'], ['japanese']], ['XJTGY', ['korean'], ['chinese']], ['LEOQX', ['indian', 'italian', 'spanish', 'mexican', 'moroccan', 'greek', 'cajun_creole', 'jamaican', 'brazilian'], ['russian', 'french', 'southern_us', 'british', 'irish']], ['OLHDN', ['italian', 'spanish', 'mexican', 'greek', 'cajun_creole', 'brazilian'], ['indian', 'moroccan', 'jamaican']], ['TIBCQ', ['italian', 'spanish', 'greek'], ['mexican', 'cajun_creole', 'brazilian']], ['HJMQG', ['italian', 'greek'], ['spanish']], ['XPECS', ['italian'], ['greek']], ['LZODD', ['mexican', 'brazilian'], ['cajun_creole']], ['RHYEL', ['mexican'], ['brazilian']], ['QGMQI', ['indian', 'moroccan'], ['jamaican']], ['CAGOZ', ['indian'], ['moroccan']], ['DSVLM', ['russian', 'southern_us', 'british', 'irish'], ['french']], ['VBOQY', ['russian'], ['southern_us', 'british', 'irish']], ['MBXKL', ['southern_us'], ['british', 'irish']], ['FSKVW', ['british'], ['irish']]]

print("Loading data ...")
train_raw, train_label_raw = utils.loadTrainData()

train_raw = [" ".join(t) for t in train_raw]

tfidf = TfidfVectorizer()
train_ft = tfidf.fit_transform(train_raw)
utils.saveModel(tfidf, "tfidf")

for (model, groupA, groupB) in tree:
    train, _, train_label = utils.dataSplit(train_ft, train_label_raw, groupA, groupB)
    train = vstack(train)
    # Label Encoding
    lb = LabelEncoder()
    oh_train_label = lb.fit_transform(train_label)

    print("Performing grid search ...")
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=0)
    grid_search.fit(train, oh_train_label)
    print(model)
    print("---------------------------")
    print(groupA)
    print(groupB)
    print("Best score: %0.5f" % grid_search.best_score_)
    print("Best param:", grid_search.best_params_)
    print()
    # for param, score in zip(grid_search.cv_results_['params'], grid_search.cv_results_["mean_test_score"]):
    #     print("Accuracy", score)
    #     print("---------------------")
    #     for k, v in param.items():
    #         print(k, ":", v)
    #     print()
    utils.saveModel(grid_search.best_estimator_, model)

