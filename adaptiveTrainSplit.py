# Import the required libraries

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from scipy.sparse import vstack
import numpy as np
import os
import pathlib
import torch
import mlp
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

tree =[['EKWHM', ['vietnamese', 'korean', 'japanese', 'thai', 'filipino', 'chinese'], ['russian', 'indian', 'italian', 'french', 'spanish', 'mexican', 'southern_us', 'moroccan', 'british', 'greek', 'cajun_creole', 'jamaican', 'brazilian', 'irish']], ['OHVSH', ['korean', 'japanese', 'filipino', 'chinese'], ['vietnamese', 'thai']], ['CDVBK', ['korean', 'japanese', 'chinese'], ['filipino']], ['NYJZU', ['japanese'], ['korean', 'chinese']], ['KCUAY', ['korean'], ['chinese']], ['PINYX', ['vietnamese'], ['thai']], ['GVNZC', ['indian', 'italian', 'spanish', 'mexican', 'moroccan', 'greek', 'cajun_creole', 'jamaican', 'brazilian'], ['russian', 'french', 'southern_us', 'british', 'irish']], ['LUSJD', ['indian', 'mexican', 'moroccan', 'jamaican', 'brazilian'], ['italian', 'spanish', 'greek', 'cajun_creole']], ['QBBVU', ['mexican', 'jamaican', 'brazilian'], ['indian', 'moroccan']], ['INVTY', ['jamaican', 'brazilian'], ['mexican']], ['BAVCM', ['jamaican'], ['brazilian']], ['SOGHT', ['indian'], ['moroccan']], ['MZMYF', ['cajun_creole'], ['italian', 'spanish', 'greek']], ['TRWQB', ['spanish'], ['italian', 'greek']], ['YFJMW', ['italian'], ['greek']], ['VIVIN', ['southern_us', 'british', 'irish'], ['russian', 'french']], ['JYZLD', ['british', 'irish'], ['southern_us']], ['UONUN', ['british'], ['irish']], ['NYIEO', ['russian'], ['french']]]
struct ={'EKWHM': ['OHVSH', 'GVNZC'], 'OHVSH': ['CDVBK', 'PINYX'], 'CDVBK': ['NYJZU', 'filipino'], 'NYJZU': ['japanese', 'KCUAY'], 'KCUAY': ['korean', 'chinese'], 'PINYX': ['vietnamese', 'thai'], 'GVNZC': ['LUSJD', 'VIVIN'], 'LUSJD': ['QBBVU', 'MZMYF'], 'QBBVU': ['INVTY', 'SOGHT'], 'INVTY': ['BAVCM', 'mexican'], 'BAVCM': ['jamaican', 'brazilian'], 'SOGHT': ['indian', 'moroccan'], 'MZMYF': ['cajun_creole', 'TRWQB'], 'TRWQB': ['spanish', 'YFJMW'], 'YFJMW': ['italian', 'greek'], 'VIVIN': ['JYZLD', 'NYIEO'], 'JYZLD': ['UONUN', 'southern_us'], 'UONUN': ['british', 'irish'], 'NYIEO': ['russian', 'french']}

print("Loading data ...")
train_raw, train_label_raw = utils.loadTrainData()

train_raw = [" ".join(t) for t in train_raw]

tfidf = TfidfVectorizer()
train = tfidf.fit_transform(train_raw)
utils.saveModel(tfidf, "tfidf")
train_label = train_label_raw

dct = {}
for (modelName, groupA, groupB) in tree:
    if modelName in dct:
        train, train_label, oh_train_label = utils.dataSplit(*dct[modelName], groupA, groupB)
    else:
        train, train_label, oh_train_label = utils.dataSplit(train, train_label, groupA, groupB)
    train = vstack(train)
    print("Performing grid search ...")
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=0)
    grid_search.fit(train, oh_train_label)

    utils.saveModel(grid_search.best_estimator_, modelName)
    tmp_dct = utils.adjustTrainingSet(train, train_label, struct[modelName], grid_search.best_estimator_)
    dct = {**dct, **tmp_dct}
    print(modelName)
    print("---------------------------")
    print(groupA, len(list(filter(lambda x: x in groupA, dct[struct[modelName][0]][1]))), len(list(filter(lambda x: x in groupA, train_label_raw))))
    print(groupB, len(list(filter(lambda x: x in groupB, dct[struct[modelName][1]][1]))), len(list(filter(lambda x: x in groupB, train_label_raw))))
    print("Best score: %0.5f" % grid_search.best_score_)
    print("Best param:", grid_search.best_params_)
    print()

