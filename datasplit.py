"""
The test set provided does not have any labels.

Therefore, only the training set will be used. The held-out test set containing 10% of the training data is extracted from the training set.
The remaining training data is folded into five to allow for five fold validation for hyperparameters tuning.
"""
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import os
import pathlib
import json
from collections import Counter

cwd = pathlib.Path(os.getcwd())

# Load original training dataset
data = json.load(open(str(cwd / "original" / "train.json")))

size = len(data)

labels = np.array([d['cuisine'].lower() for d in data])
ingredients = np.array([d['ingredients'] for d in data])

train, test, train_labels, test_labels, = train_test_split(ingredients, labels, test_size=0.1, stratify=labels)
json.dump({"data": test.tolist(), "label": test_labels.tolist()}, open(str(cwd / "data2" / "test.json"), "w+"))
json.dump({"data": train.tolist(), "label": train_labels.tolist()}, open(str(cwd / "data2" / "train.json"), "w+"))
#
# for i, (train_index, val_index) in enumerate(StratifiedKFold(n_splits=5, shuffle=True).split(train, train_labels)):
#     t_fold, tl_fold = train[train_index], train_labels[train_index]
#     v_fold, vl_fold = train[val_index], train_labels[val_index]
#
#     folddir = cwd / "fold_data" / str(i+1)
#     folddir.mkdir(exist_ok=False)
#
#     json.dump({"data": t_fold.tolist(), "label": tl_fold.tolist()}, open(str(folddir / "train.json"), "w+"))
#     json.dump({"data": v_fold.tolist(), "label": vl_fold.tolist()}, open(str(folddir / "val.json"), "w+"))
#
#
