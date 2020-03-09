from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import scipy
import numpy as np
import json
import os
from collections import defaultdict
import pathlib
import utils

import matplotlib.pyplot as plt

cwd = pathlib.Path(os.getcwd())

param_grid = ParameterGrid({
    # 'tfidf__use_idf': [True],
    # 'tfidf__norm': ['l2'],
    # 'tfidf__max_features': [None],
    # 'tfidf__max_df': [1.0, 0.9, 0.8],
    # 'tfidf__min_df': [0.0, 0.1, 0.2],
    'estimator__kernel': ['rbf'],
    'estimator__C': [2, 5],
    'estimator__gamma': ["scale"]
    # "estimator__coef0" : [0.0, 0.1, 1.0]
})

# pipeline = Pipeline([
    # ('tfidf', TfidfVectorizer()),
    # ('clf', OneVsRestClassifier(SVC())),
# ])

for fold in range(1, 2):
    train_data, train_label, val_data, val_label = utils.loadFoldData(fold)

    stopwords = utils.PMI(train_data, train_label)


    sorted_data = defaultdict(list)
    # rest_data = defaultdict(list)
    # for i, c in enumerate(train_label):
    #     sorted_data[c].append(train_data[i])
    #     for cl in set(train_label):
    #         if c != cl:
    #             rest_data[cl].append(train_data[i])
    #
    # k = "southern_us"
    # v = sorted_data[k]
    # word_count = np.array([len(t) for t in v])
    # rest_w_count = np.array([len(t) for t in rest_data[k]])
    # plt.hist(word_count, normed=True, bins=20, alpha=0.5, label=k)
    # plt.hist(rest_w_count, normed=True, bins=20, alpha=0.5, label="Rest")
    # plt.legend(loc='upper right')
    # plt.show()

    train_word_count = np.array([len(t) for t in train_data])
    val_word_count = np.array([len(t) for t in val_data])

    train = [" ".join(t) for t in train_data]
    val = [" ".join(t) for t in val_data]

    train_char_count = (np.array([len(t) for t in train]) - train_word_count + 1)/train_word_count
    val_char_count = (np.array([len(t) for t in val]) - val_word_count + 1)/val_word_count

    # train_word_count = normalize(train_word_count.reshape(-1, 1))
    # val_word_count = normalize(val_word_count.reshape(-1, 1))
    #
    # train_char_count = normalize(train_char_count.reshape(-1, 1))
    # val_char_count = normalize(val_char_count.reshape(-1, 1))

    # plt.hist(train_char_count, normed=True, bins=100)
    # plt.ylabel('Probability');
    # plt.show()
    #
    # plt.hist(train_word_count, normed=True, bins=100)
    # plt.ylabel('Probability');
    # plt.show()

    train = [" ".join(t) for t in utils.filterByTotalFreq(train, 0.999, 0.2)]

    # Get PMI features
    PMI = utils.PMI(train_data, train_label)
    train_pmi_score = [sum([utils.getPMIScore(word, PMI) for word in t]) for t in train_data]
    val_pmi_score = [sum([utils.getPMIScore(word, PMI) for word in t]) for t in val_data]

    train_pmi_score = normalize(train_pmi_score)
    val_pmi_score = normalize(val_pmi_score)
    #
    # tfidf = TfidfVectorizer()
    # train_ft = tfidf.fit_transform(train)
    # val_ft = tfidf.transform(val)
    #
    # # train_ft = scipy.sparse.hstack((np.array(train_pmi_score), train_word_count, train_char_count, train_ft))
    # # val_ft = scipy.sparse.hstack((np.array(val_pmi_score), val_word_count, val_char_count, val_ft))
    #
    # # train_ft = normalize(train_ft)
    # # val_ft = normalize(val_ft)
    #
    # lb = LabelEncoder()
    # oh_train_label = lb.fit_transform(train_label)
    #
    # for param in param_grid:
    #     print("Training", param)
    #     model = OneVsRestClassifier(SVC())
    #     model.set_params(**param)
    #     model.fit(train_ft, oh_train_label)
    #
    #     val_out = model.predict(val_ft)
    #     val_pred = lb.inverse_transform(val_out)
    #
    #     classes = list(lb.classes_)
    #     cm = np.zeros((len(classes), len(classes)))
    #
    #     for ci, c in enumerate(classes):
    #         for index, truth in enumerate(val_label):
    #             if truth == c:
    #                 cm[ci][classes.index(val_pred[index])] += 1
    #
    #     print(classes)
    #     print(cm)
    #
    #     acc = np.sum(val_label == val_pred) / len(val_pred)
    #     print("Fold", fold, param)
    #     print("Accuracy:", acc)
    #     print()