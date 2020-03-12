# Import the required libraries

from scipy.sparse import vstack
from collections import OrderedDict
import os
import pathlib
import pandas as pd

import utils

cwd = pathlib.Path(os.getcwd())

class Tree:
    folder = r"models\\adj_bdtsvm"
    treestruct = OrderedDict(
        {'FNSHG': ['HLKHN', 'ALDWR'], 'HLKHN': ['EPFMO', 'LVOAD'], 'EPFMO': ['KFJLQ', 'NWGJE'], 'KFJLQ': ['CROWL', 'southern_us'], 'CROWL': ['british', 'irish'], 'NWGJE': ['french', 'russian'], 'LVOAD': ['CSAIM', 'TFTND'], 'CSAIM': ['spanish', 'LVBVD'], 'LVBVD': ['greek', 'italian'], 'TFTND': ['SNQRU', 'ZYWWZ'], 'SNQRU': ['VKZKU', 'ZOSXZ'], 'VKZKU': ['mexican', 'brazilian'], 'ZOSXZ': ['jamaican', 'cajun_creole'], 'ZYWWZ': ['indian', 'moroccan'], 'ALDWR': ['SLCMH', 'UJSUP'], 'SLCMH': ['DTMMR', 'filipino'], 'DTMMR': ['japanese', 'FKWSQ'], 'FKWSQ': ['chinese', 'korean'], 'UJSUP': ['thai', 'vietnamese']}
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
            self.model = utils.loadModel(model, Tree.folder)
            self.zero = Tree(self.treestruct[model][0])
            self.one = Tree(self.treestruct[model][1])

    def predict(self, data):
        if self.model is not None:
            if self.model.predict(data) == 0:
                return self.zero.predict(data)
            else:
                return self.one.predict(data)
        else:
            return self.label


# Load Test Data
print("Preparing Test Data ...")
test_id, test = utils.loadPreprocessTest()

tfidf = utils.loadModel("tfidf", Tree.folder)
test_ft = tfidf.transform(test)
test_ft = vstack(test_ft)

btsvm = Tree()
preds = [btsvm.predict(t) for t in test_ft]

print ("Generate Submission File ... ")
sub = pd.DataFrame({'id': test_id, 'cuisine': preds}, columns=['id', 'cuisine'])
sub.to_csv('submission.csv', index=False)