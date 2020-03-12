from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import os
import pathlib
import utils
import ltsvm.twinsvm as TSVM

"""
Twin SVM One-Versus-One 

Model size is too big to upload (15 Gb). 
"""

cwd = pathlib.Path(os.getcwd())

parameters = {
        'clf__kernel': ['RBF'],
        'clf__gamma': [0.05, 0.1, 0.25, 1],
        'clf__C1': [0.1, 0.5, 1, 5],
        'clf__C2': [0.1, 0.5, 1, 5],
    }

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=10)),
    ('clf', TSVM.OVO_TSVM()),
])

train, train_label = utils.loadPreprocessed()

# Label Encoding
lb = LabelEncoder()
oh_train_label = lb.fit_transform(train_label)

print("Performing grid search ...")
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=1, verbose=10)
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

utils.saveModel(grid_search.best_estimator_, "tsvmOvO")