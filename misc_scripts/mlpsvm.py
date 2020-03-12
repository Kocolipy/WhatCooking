import matplotlib.pyplot as plt
import itertools
from collections import defaultdict, Counter
import torch.nn as nn
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader
import numpy as np

import utils

class MyDataset(Dataset):
    def __init__(self, data, labels):
        super(MyDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # label = np.zeros(20)
        # label[self.labels[index]] = 1
        return self.data[index], torch.tensor(self.labels[index])

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(20, 40),
            # nn.BatchNorm1d(40),
            nn.LeakyReLU(),
            nn.Linear(40, 40),
            # nn.BatchNorm1d(40),
            nn.LeakyReLU(0.2),
            nn.Linear(40, 20)
        )
    def forward(self, x):
        return self.fc(x)

# Load preprocessing model, classifier and label encoder
print("Loading Models ...")
# preprocess, wc_model, _ = utils.loadModel("with_wc")
# _, model, lb = utils.loadModel("with_negated_wc")
preprocess, model, lb = utils.loadModel("best_starting")
_, susmodel, _ = utils.loadModel("southern_us")

# Load Test Data
print("Preparing validation data ...")
test, test_label = utils.loadFoldValData(1)

# test, test_label = utils.loadTestData()
# test_word_count = np.array([len(t) for t in test])
# test_word_count = preprocess.transform(test_word_count.reshape(-1, 1))

test_ft = [" ".join(t).lower() for t in test]
test_ft = preprocess.transform(test_ft).astype('float16')
# test_ft = hstack((test_word_count, test_ft))

sorted_data = defaultdict(list)
for i, c in enumerate(test_label):
    sorted_data[c].append(i)

scores = torch.tensor(np.array([m.decision_function(test_ft) for m in model.estimators_])).T.float()

# model = MLP()
# checkpoint = torch.load(str(20))
# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval()
#
# output = model(scores)
# preds = np.argsort(-torch.nn.Softmax(dim=1)(output).detach().numpy(), axis=1).T[0]
# test_pred = lb.inverse_transform(preds)
# acc = np.sum(test_label == test_pred) / len(test_pred)
# print("Accuracy:", acc)
# print()


# dataset = MyDataset(scores, lb.transform(test_label))
# dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

# print("Training ...")
# if __name__ == '__main__':
#     torch.multiprocessing.freeze_support()
#     device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#
#     model = MLP()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters())
#
#     for epoch in range(100):
#         print("Epoch", epoch)
#         model.train()
#         optimizer.zero_grad()
#         for data, label in dataloader:
#             pred = model(data)
#
#             loss = criterion(pred, label)
#
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#
#
#         torch.save({
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#         }, str(epoch))
#
#         output = model(scores)
#         preds = np.argsort(-torch.nn.Softmax(dim=1)(output).detach().numpy(), axis=1).T[0]
#         test_pred = lb.inverse_transform(preds)
#         acc = np.sum(test_label == test_pred) / len(test_pred)
#         print("Accuracy:", acc)
#         print()

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

# SUSft = vstack(sorted_data["southern_us"])

# WCSUSvAll = wc_model.estimators_[16]
# SUSvAll = model.estimators_[16]

# for c in set(test_label):
#     fts = vstack([test_ft.getrow(i) for i in sorted_data[c]])
#     wc_pred = len(np.nonzero(WCSUSvAll.predict(fts))[0])
#     pred = len(np.nonzero(SUSvAll.predict(fts))[0])
#     print(c, wc_pred, pred)

# count = 0
# wrong = []
# for i in np.nonzero(prediction)[0]:
#     if test_label[i] == "southern_us":
#         count += 1
#     else:
#         wrong.append(test_label[i])

# model.estimators_[16] = WCSUSvAll





# Predictions
print("Predicting ...")
test_out = model.predict(test_ft)
test_pred = lb.inverse_transform(test_out)

# Calculate Accuracy
acc = np.sum(test_label == test_pred) / len(test_pred)
print(acc)



classes = list(lb.classes_)
cm = np.zeros((len(classes), len(classes)))

for ci, c in enumerate(classes):
    for index, truth in enumerate(test_label):
        if truth == c:
            cm[ci][classes.index(test_pred[index])] += 1


# groups = np.array([
#     [48,21,75,3541],
#     [3,5,5,958],
#     [4480,5882,394,19],
#     [10,0,3,38]]
# )
#
plot_confusion_matrix(cm, "Confusion Matrix", classes)
