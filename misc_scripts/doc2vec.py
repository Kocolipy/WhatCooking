from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np

import os
import pathlib
import utils

cwd = pathlib.Path(os.getcwd())

parameters = {
    'estimator__kernel': ['rbf'],
    'estimator__C': [1, 2, 5],
    'estimator__gamma': ["scale"],
}

train_data, train_label = utils.loadOriginalTrain()
train_data = [utils.preprocess(t) for t in train_data]

lb = LabelEncoder()
oh_train_label = lb.fit_transform(train_label)

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_data)]
d2v_model = Doc2Vec(documents, vector_size=20, min_count=1, workers=4, epochs=25)

train_ft = np.array([d2v_model.infer_vector([t]) for t in train_data])
train_ft = normalize(train_ft)

print("Grid Searching ...")
grid_search = GridSearchCV(OneVsRestClassifier(SVC(), n_jobs=4), parameters, cv=5, n_jobs=1, verbose=10)
grid_search.fit(train_ft, oh_train_label)
print("Best score: %0.5f" % grid_search.best_score_)
print("Best param:", grid_search.best_params_)
print()
for param, score in zip(grid_search.cv_results_['params'], grid_search.cv_results_["mean_test_score"]):
    print("Accuracy", score)
    print("---------------------")
    for k, v in param.items():
        print(k, ":", v)
    print()

utils.saveModel({"preprocess": d2v_model, "model": grid_search.best_estimator_, "labelencoder": lb}, "D2V-20-OvR")