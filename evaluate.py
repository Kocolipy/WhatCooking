import matplotlib.pyplot as plt
import itertools
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import utils
from sklearn.metrics import confusion_matrix

"""
Perform evaluation with confusion matrix for more insights

Only valid for models trained using a held-out test data (use datasplit.py)
"""

# Load preprocessing model, classifier and label encoder
print("Loading Models ...")
model_data = utils.loadModel("best_svmOvR_half", "models")
model = model_data["model"]
lb = model_data["labelencoder"]

# Load Test Data
print("Preparing Test Data ...")
test_ft, test_label = utils.loadTestData()
test_ft = test_ft
test_label = test_label
test_ft = [utils.preprocess(t) for t in test_ft]

# #Without preprocessing
# test_ft, test_label = utils.loadTestData()
# test_ft = [" ".join(t).lower() for t in test_ft]


# Predictions
print("Predicting ...")
test_out = model.predict(test_ft)
test_pred = lb.inverse_transform(test_out)

# Generate confusion matrix
def plot_confusion_matrix(pred, labels, classes):
    plt.figure(figsize=(10, 10))

    cm = confusion_matrix(labels, pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = plt.imshow(cm_normalized, interpolation='nearest', cmap="Blues")
    plt.title("")

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{0}".format(int(cm[i, j])), fontname="Calibri", fontsize=12,
                 horizontalalignment="center",  va="center",
                 color="white" if i == j else "black")

    plt.colorbar(im, fraction=0.05)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Ground Truth')
    plt.xlabel('Predictions')
    plt.show()

# Calculate Accuracy
acc = np.sum(test_label == test_pred) / len(test_pred)
print("Accuracy", acc)

classes = list(lb.classes_)
cm = np.zeros((len(classes), len(classes)))

for ci, c in enumerate(classes):
    for index, truth in enumerate(test_label):
        if truth == c:
            cm[ci][classes.index(test_pred[index])] += 1
print(cm)


# plot_confusion_matrix(test_pred, test_label, classes)

print()
