# Import the required libraries

import numpy as np
import os
import pathlib

import utils

""""
Evaluate the individual binary classifiers involved in the BTSVM
"""
cwd = pathlib.Path(os.getcwd())

folder = "ABTSVM\\tf-idf_rbf_models"
tree = [['WQYHB', ['vietnamese', 'korean', 'japanese', 'thai', 'filipino', 'chinese'], ['russian', 'indian', 'italian', 'french', 'spanish', 'mexican', 'southern_us', 'moroccan', 'british', 'greek', 'cajun_creole', 'jamaican', 'brazilian', 'irish']], ['WFGKZ', ['korean', 'japanese', 'filipino', 'chinese'], ['vietnamese', 'thai']], ['KGDCQ', ['korean', 'japanese', 'chinese'], ['filipino']], ['KHJHA', ['japanese'], ['korean', 'chinese']], ['RZMQO', ['korean'], ['chinese']], ['HUJVM', ['vietnamese'], ['thai']], ['RIWSP', ['indian', 'italian', 'spanish', 'mexican', 'moroccan', 'greek', 'cajun_creole', 'jamaican', 'brazilian'], ['russian', 'french', 'southern_us', 'british', 'irish']], ['VZWJV', ['indian', 'mexican', 'moroccan', 'jamaican', 'brazilian'], ['italian', 'spanish', 'greek', 'cajun_creole']], ['VZVLW', ['mexican', 'jamaican', 'brazilian'], ['indian', 'moroccan']], ['AHYVM', ['jamaican', 'brazilian'], ['mexican']], ['HTWPC', ['jamaican'], ['brazilian']], ['DMWSF', ['indian'], ['moroccan']], ['FWDAK', ['cajun_creole'], ['italian', 'spanish', 'greek']], ['GQUYB', ['spanish'], ['italian', 'greek']], ['FCFBN', ['italian'], ['greek']], ['IPLEL', ['southern_us', 'british', 'irish'], ['russian', 'french']], ['MDIDC', ['british', 'irish'], ['southern_us']], ['HINRA', ['british'], ['irish']], ['IHCDE', ['russian'], ['french']]]

# folder = "tf-idf_rbf_models"
# tree = [['EKWHM', ['vietnamese', 'korean', 'japanese', 'thai', 'filipino', 'chinese'], ['russian', 'indian', 'italian', 'french', 'spanish', 'mexican', 'southern_us', 'moroccan', 'british', 'greek', 'cajun_creole', 'jamaican', 'brazilian', 'irish']], ['OHVSH', ['korean', 'japanese', 'filipino', 'chinese'], ['vietnamese', 'thai']], ['CDVBK', ['korean', 'japanese', 'chinese'], ['filipino']], ['NYJZU', ['japanese'], ['korean', 'chinese']], ['KCUAY', ['korean'], ['chinese']], ['PINYX', ['vietnamese'], ['thai']], ['GVNZC', ['indian', 'italian', 'spanish', 'mexican', 'moroccan', 'greek', 'cajun_creole', 'jamaican', 'brazilian'], ['russian', 'french', 'southern_us', 'british', 'irish']], ['LUSJD', ['indian', 'mexican', 'moroccan', 'jamaican', 'brazilian'], ['italian', 'spanish', 'greek', 'cajun_creole']], ['QBBVU', ['mexican', 'jamaican', 'brazilian'], ['indian', 'moroccan']], ['INVTY', ['jamaican', 'brazilian'], ['mexican']], ['BAVCM', ['jamaican'], ['brazilian']], ['SOGHT', ['indian'], ['moroccan']], ['MZMYF', ['cajun_creole'], ['italian', 'spanish', 'greek']], ['TRWQB', ['spanish'], ['italian', 'greek']], ['YFJMW', ['italian'], ['greek']], ['VIVIN', ['southern_us', 'british', 'irish'], ['russian', 'french']], ['JYZLD', ['british', 'irish'], ['southern_us']], ['UONUN', ['british'], ['irish']], ['NYIEO', ['russian'], ['french']]]

print("Loading data ...")
test_data, test_labels = utils.loadTestData()
test_data = [" ".join(t) for t in test_data]

# # For constant tf-idf
# tfidf = utils.loadModelSVM("tfidf", "tf-idf_rbf_models")
# test = tfidf.transform(test_data)
# test = vstack(test)

# For adaptive tf-idf
test = test_data

for (model, groupA, groupB) in tree:
    test_f, _, oh_test_label = utils.dataSplit(test, test_labels, groupA, groupB)
    # test_f = vstack(test_f)

    svm = utils.loadModelSVM(model, folder)
    preds = svm.predict(test_f)
    print(model)
    print(groupA)
    print(groupB)
    print(sum(np.array(preds) == np.array(oh_test_label))/len(preds))
    print()
