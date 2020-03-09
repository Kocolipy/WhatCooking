# Import the required libraries

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pandas as pd
from scipy.sparse import vstack
from collections import OrderedDict
import matplotlib.pyplot as plt
import itertools
import numpy as np
import os
import pathlib
import torch

import mlp
import utils

cwd = pathlib.Path(os.getcwd())

class Tree:
    folder = "adjustedTrain"
    tfidf = utils.loadModelSVM("tfidf", folder)
    treestruct = OrderedDict(
        {'EKWHM': ['OHVSH', 'GVNZC'], 'OHVSH': ['CDVBK', 'PINYX'], 'CDVBK': ['NYJZU', 'filipino'], 'NYJZU': ['japanese', 'KCUAY'], 'KCUAY': ['korean', 'chinese'], 'PINYX': ['vietnamese', 'thai'], 'GVNZC': ['LUSJD', 'VIVIN'], 'LUSJD': ['QBBVU', 'MZMYF'], 'QBBVU': ['INVTY', 'SOGHT'], 'INVTY': ['BAVCM', 'mexican'], 'BAVCM': ['jamaican', 'brazilian'], 'SOGHT': ['indian', 'moroccan'], 'MZMYF': ['cajun_creole', 'TRWQB'], 'TRWQB': ['spanish', 'YFJMW'], 'YFJMW': ['italian', 'greek'], 'VIVIN': ['JYZLD', 'NYIEO'], 'JYZLD': ['UONUN', 'southern_us'], 'UONUN': ['british', 'irish'], 'NYIEO': ['russian', 'french']}
            .items())
    labels = ["brazilian", "cajun_creole", "mexican", "southern_us", "spanish", "italian", "greek", "french", "russian",
              "thai", "vietnamese", "british", "jamaican", "irish", "indian", "moroccan", "chinese", "filipino", "japanese", "korean"]

    def __init__(self, model=None):
        if model is None:
            model = list(Tree.treestruct.keys())[0]
        self.model = None
        self.label = None
        if model in self.labels:
            self.label = model
        else:
            self.model = utils.loadModelSVM(model, Tree.folder)
            self.zero = Tree(self.treestruct[model][0])
            self.one = Tree(self.treestruct[model][1])

    def predict(self, data):
        if self.model is not None:
            ft = Tree.tfidf.transform([data]) if len(self.model.steps) == 1 else [data]
            if self.model.predict(ft) == 0:
                return self.zero.predict(data)
            else:
                return self.one.predict(data)
        else:
            return self.label


def plot_confusion_matrix(cm, title, labels):
    cmap = plt.get_cmap('Blues')

    fig, ax = plt.subplots()
    plt.imshow(cm, cmap=cmap)
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=80, ha="right")
    plt.yticks(tick_marks, labels)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:.0f}".format(cm[i, j]), fontname="Calibri", fontsize=16,
                 horizontalalignment="center",  va="center",
                 color="white" if cm[i, j] > 50 else "black")

    plt.tight_layout()
    plt.title(title)
    plt.ylabel('Ground Truth Labels')
    plt.xlabel("Predicted Labels")
    plt.show()


print("Loading data ...")
test_id, test_data = utils.loadOfficialTestData()
# test_data, test_labels = utils.loadTestData()
test = [" ".join(t) for t in test_data]

btsvm = Tree()

preds = [btsvm.predict(t) for t in test]

# print(sum(np.array(preds) == np.array(test_labels))/len(preds))
#
# classes = list(sorted(set(test_labels)))
# print(classes)
# cm = np.zeros((len(classes), len(classes)))
#
# for ci, c in enumerate(classes):
#     for index, truth in enumerate(test_labels):
#         if truth == c:
#             cm[ci][classes.index(preds[index])] += 1
#
# print(cm)
#
# plot_confusion_matrix(cm, "Confusion Matrix", classes)
# print()

print ("Generate Submission File ... ")
sub = pd.DataFrame({'id': test_id, 'cuisine': preds}, columns=['id', 'cuisine'])
sub.to_csv('submission.csv', index=False)