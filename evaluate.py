import matplotlib.pyplot as plt
import itertools
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

import numpy as np

import utils
# Generate confusion matrix
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


# Load preprocessing model, classifier and label encoder
print("Loading Models ...")
model, lb = utils.loadModel("DFmin-SVMOvR")

# Load Test Data
print("Preparing validation data ...")
test_id, test_data = utils.loadOfficialTestData()
# test, test_label = utils.loadTestData()


# test_ft = [" ".join(t).lower() for t in test_data]
test_ft = [utils.preprocess(t) for t in test_data]

# Predictions
print("Predicting ...")
test_out = model.predict(test_ft)

print("predicted")
test_pred = lb.inverse_transform(test_out)

print ("Generate Submission File ... ")
sub = pd.DataFrame({'id': test_id, 'cuisine': test_pred}, columns=['id', 'cuisine'])
sub.to_csv('submission.csv', index=False)

# # Calculate Accuracy
# acc = np.sum(test_label == test_pred) / len(test_pred)
# print(acc)
#
# classes = list(lb.classes_)
# cm = np.zeros((len(classes), len(classes)))
#
# for ci, c in enumerate(classes):
#     for index, truth in enumerate(test_label):
#         if truth == c:
#             cm[ci][classes.index(test_pred[index])] += 1
#
# plot_confusion_matrix(cm, "Confusion Matrix", classes)
# print()
