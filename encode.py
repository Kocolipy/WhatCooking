from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import scipy
import torch
import numpy as np
import json
import mlp
import os
import pathlib
import utils

cwd = pathlib.Path(os.getcwd())

param_grid = ParameterGrid({
    # 'tfidf__use_idf': [True],
    # 'tfidf__norm': ['l2'],
    # 'tfidf__max_features': [None],
    # 'tfidf__max_df': [1.0, 0.9, 0.8],
    # 'tfidf__min_df': [0.0, 0.1, 0.2],
    'estimator__kernel': ['rbf'],
    'estimator__C': [10, 100],
    'estimator__gamma': ["scale"],
})

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    for fold in range(1, 2):
        train_data, train_label, val_data, val_label = utils.loadFoldData(fold)

        train = [" ".join(t) for t in train_data]
        val = [" ".join(t) for t in val_data]

        tfidf = TfidfVectorizer()
        train_ft = tfidf.fit_transform(train)
        val_ft = tfidf.transform(val)

        model = mlp.MLP(train_ft.shape[1])

        # Load from previous checkpoint
        ckpt_path = cwd / "ae_ckpt"
        ckpt_path.mkdir(exist_ok=True)
        checkpoint = torch.load(str(ckpt_path / "100"))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print("Loading Checkpoint 100")


        # convert to dense vectors
        train_ft = [torch.sparse.FloatTensor(torch.from_numpy(t.indices).unsqueeze(0).long(),
                                 torch.from_numpy(t.data).float(), [train_ft.shape[1]]).to_dense() for t in train_ft]
        val_ft = [torch.sparse.FloatTensor(torch.from_numpy(t.indices).unsqueeze(0).long(),
                                 torch.from_numpy(t.data).float(), [val_ft.shape[1]]).to_dense() for t in val_ft]

        with torch.no_grad():
            # Convert to 128d vectors
            train_ft = model(torch.stack(train_ft))
            val_ft = model(torch.stack(val_ft))

        # train_ft = normalize(train_ft)
        # val_ft = normalize(val_ft)

        lb = LabelEncoder()
        oh_train_label = lb.fit_transform(train_label)

        for param in param_grid:
            print("Training", param)
            model = OneVsRestClassifier(SVC())
            model.set_params(**param)
            model.fit(train_ft, oh_train_label)

            val_out = model.predict(val_ft)
            val_pred = lb.inverse_transform(val_out)

            acc = np.sum(val_label == val_pred) / len(val_pred)
            print("Fold", fold, param)
            print("Accuracy:", acc)
            print()