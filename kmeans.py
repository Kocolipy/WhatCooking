from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, SpectralClustering
from collections import defaultdict
from scipy.sparse import vstack
import numpy as np
from random import choices
from string import ascii_uppercase

import os
import pathlib

import utils

cwd = pathlib.Path(os.getcwd())

print("Loading data ...")
train, train_label = utils.loadPreprocessed()

# Combine training data into a giant class document
grouped_docs = defaultdict(list)
for i, c in enumerate(train_label):
    grouped_docs[c] += [train[i]]

train = {}
for k, v in grouped_docs.items():
    train[k] = " ".join(v)

class Tree:
    # kmeans = KMeans(n_clusters=2)
    kmeans = SpectralClustering(n_clusters=2, gamma=1, random_state=0)
    data = train
    all_classes = set(["chinese", "filipino", "japanese", "korean", "thai", "vietnamese", "brazilian", "cajun_creole",
                   "mexican", "southern_us", "spanish", "italian", "greek", "french", "russian", "british", "jamaican",
                   "irish", "indian", "moroccan"])

    def __init__(self, classes):
        self.node = "".join(choices(ascii_uppercase, k=5))
        self.isLeaf = False
        self.classes = classes
        if len(classes) == 1:
            self.node = classes[0]
            self.isLeaf = True
        elif len(classes) == 2:
            self.left = Tree(np.array([classes[0]]))
            self.right = Tree(np.array([classes[1]]))
        else:
            exclude = list(Tree.all_classes - set(classes))
            tfidf = TfidfVectorizer()
            data = Tree.data.copy()

            # Adaptive tf-idf
            # for i in exclude:
            #     del data[i]

            tfidf.fit(data.values())

            train_ft = {}
            for k, v in data.items():
                train_ft[k] = tfidf.transform([data[k]])

            # # Constant tf-idf
            for i in exclude:
                del train_ft[i]

            results = Tree.kmeans.fit(vstack(train_ft.values())).labels_

            self.left = Tree(np.array(list(filter(lambda x: x[0] == 0, zip(results, list(train_ft.keys()))))).T[1])
            self.right = Tree(np.array(list(filter(lambda x: x[0] == 1, zip(results, list(train_ft.keys()))))).T[1])

    def getStruct(self):
        struct = [[self.node, self.left.classes.tolist(), self.right.classes.tolist()]]
        if not self.left.isLeaf:
            struct += self.left.getStruct()
        if not self.right.isLeaf:
            struct += self.right.getStruct()
        return struct

    def getBTSVMTree(self):
        struct = {self.node : [self.left.node, self.right.node]}
        if not self.left.isLeaf:
            struct = {**struct, **self.left.getBTSVMTree()}
        if not self.right.isLeaf:
            struct = {**struct, **self.right.getBTSVMTree()}
        return struct

    def __str__(self):
        if self.isLeaf:
            return self.node
        else:
            left = str(self.left)
            right = str(self.right)
            return "({0}, {1})".format(left,right)

tree = Tree(["chinese", "filipino", "japanese", "korean", "thai", "vietnamese", "brazilian", "cajun_creole",
                   "mexican", "southern_us", "spanish", "italian", "greek", "french", "russian", "british", "jamaican",
                   "irish", "indian", "moroccan"])

outputdivision = tree.getStruct()
treestruct = tree.getBTSVMTree()
print(tree)
print(outputdivision)
print(treestruct)
