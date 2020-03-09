from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import scipy
import numpy as np
import json
import os
import pathlib
import utils
import ltsvm.twinsvm as tsvm

cwd = pathlib.Path(os.getcwd())

param_grid = ParameterGrid({
    # 'tfidf__use_idf': [True],
    # 'tfidf__norm': ['l2'],
    # 'tfidf__max_features': [None],
    # 'tfidf__max_df': [1.0, 0.9, 0.8],
    # 'tfidf__min_df': [0.0, 0.1, 0.2],
    # 'estimator__kernel': ['rbf'],
    # 'estimator__C': [2],
    # 'estimator__gamma': ["scale"]
    # "estimator__coef0" : [0.0, 0.1, 1.0]
})

print("Loading data ...")
train, train_label = utils.loadPreprocessed()

tfidf = TfidfVectorizer(min_df=10)

tfidf.fit_transform(train)
print(len(tfidf.vocabulary_))

# kernel = 'RBF'
#
# tsvm_model = tsvm.OVO_TSVM(kernel=kernel)
#
# c_range = {'C1': [float(2**i) for i in range(-6, 5)],
#            'C2': [float(2**i) for i in range(-6, 5)]}
#
# gamma_range = {'gamma': [float(2**i) for i in range(-8, 3)]} if kernel == 'RBF' else {}
#
# param_range = {**c_range, **gamma_range}

# pipeline = Pipeline([
    # ('tfidf', TfidfVectorizer()),
    # ('clf', OneVsRestClassifier(SVC())),
# ])
#
for fold in range(1, 6):
#     train_data, train_label, val_data, val_label = utils.loadFoldData(fold)

    # train_label = [1 if t == "southern_us" else 0 for t in train_label]

    # train_word_count = np.array([len(t) for t in train_data])
    # val_word_count = np.array([len(t) for t in val_data])

    # train_word_count = np.array([0 for t in train_data])
    # val_word_count = np.array([0 for t in val_data])

    # stopwords = utils.PMI(train_data, train_label, 1.5)
    #
    # train = [" ".join(t) for t in train_data]
    #
    #
    # val = [" ".join(t) for t in val_data]
    #
    # train_char_count = (np.array([len(t) for t in train]) - train_word_count + 1)/train_word_count
    # val_char_count = (np.array([len(t) for t in val]) - val_word_count + 1)/val_word_count
    #
    # minmaxScaler = MinMaxScaler()
    # train_word_count = minmaxScaler.fit_transform(train_word_count.reshape(-1,1))
    # val_word_count = minmaxScaler.fit_transform(val_word_count.reshape(-1,1))

    # train_char_count = normalize(train_char_count.reshape(-1, 1))
    # val_char_count = normalize(val_char_count.reshape(-1, 1))

    # train = [" ".join(t) for t in utils.filterByTotalFreq(train, 0.999, 0.2)]

    # # Get PMI features
    # PMI = utils.PMI(train_data, train_label)
    # train_pmi_score = [sum([utils.getPMIScore(word, PMI) for word in t]) for t in train_data]
    # val_pmi_score = [sum([utils.getPMIScore(word, PMI) for word in t]) for t in val_data]
    #
    # train_pmi_score = normalize(train_pmi_score)
    # val_pmi_score = normalize(val_pmi_score)

    tfidf = TfidfVectorizer()
    train_ft = tfidf.fit_transform(train)
    val_ft = tfidf.transform(val)

    # train_ft = scipy.sparse.hstack((train_word_count, train_ft))
    # val_ft = scipy.sparse.hstack((val_word_count, val_ft))

    # train_ft = scipy.sparse.hstack((np.array(train_pmi_score), train_word_count, train_char_count, train_ft))
    # val_ft = scipy.sparse.hstack((np.array(val_pmi_score), val_word_count, val_char_count, val_ft))

    # train_ft = normalize(train_ft)
    # val_ft = normalize(val_ft)

    lb = LabelEncoder()
    oh_train_label = lb.fit_transform(train_label)

    tsvm = {'Epsilon1': 0.1, 'Epsilon2': 0.1, 'C1': 1, 'C2': 1, 'kernel_type': 3, 'kernel_param': 2, 'fuzzy': 0}
    # for param in param_grid:
    # print("Training", param)
    clf = TVSVM.TwinSVMClassifier(**tsvm)
    model = OneVsRestClassifier(clf)
    # model = SVC()
    # model.set_params(**param)
    model.fit(train_ft, oh_train_label)

    val_out = model.predict(val_ft)
    val_pred = lb.inverse_transform(val_out)

        # utils.saveModel({"preprocess": tfidf, "model":model, "labelencoder":lb}, "southern_us")

        # classes = list(lb.classes_)
        # cm = np.zeros((len(classes), len(classes)))
        #
        # for ci, c in enumerate(classes):
        #     for index, truth in enumerate(val_label):
        #         if truth == c:
        #             cm[ci][classes.index(val_pred[index])] += 1
        #
        # print(classes)

        # for i, c in enumerate(classes):
        #     print("Class:", c)
        #     for index, v in enumerate(cm[i]):
        #         if v != 0:
        #             print(classes[index], v)
        # print(cm)

    acc = np.sum(val_label == val_pred) / len(val_pred)
    # print("Fold", fold, param)
    print("Accuracy:", acc)
    print()