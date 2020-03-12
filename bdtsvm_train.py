from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from scipy.sparse import vstack
import os
import pathlib
import utils

cwd = pathlib.Path(os.getcwd())

parameters = {
        'kernel': ['rbf'],
        'C': [1,  2, 5, 10],
        'gamma': ["scale", 1.1, 0.9, 1, 0.1, 1.5],
    }

tree =[['FNSHG', ['greek', 'southern_us', 'indian', 'jamaican', 'spanish', 'italian', 'mexican', 'british', 'cajun_creole', 'brazilian', 'french', 'irish', 'moroccan', 'russian'], ['filipino', 'chinese', 'thai', 'vietnamese', 'japanese', 'korean']],['HLKHN', ['southern_us', 'british', 'french', 'irish', 'russian'], ['greek', 'indian', 'jamaican', 'spanish', 'italian', 'mexican', 'cajun_creole', 'brazilian', 'moroccan']], ['EPFMO', ['southern_us', 'british', 'irish'], ['french', 'russian']], ['KFJLQ', ['british', 'irish'], ['southern_us']], ['CROWL', ['british'], ['irish']], ['NWGJE', ['french'], ['russian']], ['LVOAD', ['greek', 'spanish', 'italian'], ['indian', 'jamaican', 'mexican', 'cajun_creole', 'brazilian', 'moroccan']], ['CSAIM', ['spanish'], ['greek', 'italian']], ['LVBVD', ['greek'], ['italian']], ['TFTND', ['jamaican', 'mexican', 'cajun_creole', 'brazilian'], ['indian', 'moroccan']], ['SNQRU', ['mexican', 'brazilian'], ['jamaican', 'cajun_creole']], ['VKZKU', ['mexican'], ['brazilian']], ['ZOSXZ', ['jamaican'], ['cajun_creole']], ['ZYWWZ', ['indian'], ['moroccan']], ['ALDWR', ['filipino', 'chinese', 'japanese', 'korean'], ['thai', 'vietnamese']], ['SLCMH', ['chinese', 'japanese', 'korean'], ['filipino']], ['DTMMR', ['japanese'], ['chinese', 'korean']], ['FKWSQ', ['chinese'], ['korean']], ['UJSUP', ['thai'], ['vietnamese']]]
struct ={'FNSHG': ['HLKHN', 'ALDWR'], 'HLKHN': ['EPFMO', 'LVOAD'], 'EPFMO': ['KFJLQ', 'NWGJE'], 'KFJLQ': ['CROWL', 'southern_us'], 'CROWL': ['british', 'irish'], 'NWGJE': ['french', 'russian'], 'LVOAD': ['CSAIM', 'TFTND'], 'CSAIM': ['spanish', 'LVBVD'], 'LVBVD': ['greek', 'italian'], 'TFTND': ['SNQRU', 'ZYWWZ'], 'SNQRU': ['VKZKU', 'ZOSXZ'], 'VKZKU': ['mexican', 'brazilian'], 'ZOSXZ': ['jamaican', 'cajun_creole'], 'ZYWWZ': ['indian', 'moroccan'], 'ALDWR': ['SLCMH', 'UJSUP'], 'SLCMH': ['DTMMR', 'filipino'], 'DTMMR': ['japanese', 'FKWSQ'], 'FKWSQ': ['chinese', 'korean'], 'UJSUP': ['thai', 'vietnamese']}

print("Loading data ...")
train_raw, train_label_raw = utils.loadPreprocessed()

tfidf = TfidfVectorizer(min_df=10)
# svd = TruncatedSVD(n_components=2000)
train = tfidf.fit_transform(train_raw)
# train = svd.fit_transform(train)
utils.saveModel(tfidf, "tfidf")
# utils.saveModel(svd, "tfidf")
train_label = train_label_raw

for (modelName, groupA, groupB) in tree:
    train_ft, _, oh_train_label = utils.dataSplit(train, train_label_raw, groupA, groupB)
    train_ft = vstack(train_ft)
    print("Performing grid search ...")
    grid_search = GridSearchCV(SVC(), parameters, cv=5, n_jobs=8, verbose=0)
    grid_search.fit(train_ft, oh_train_label)

    utils.saveModel(grid_search.best_estimator_, modelName)
    print(modelName)
    print("---------------------------")
    print(groupA)
    print(groupB)
    print("Best score: %0.5f" % grid_search.best_score_)
    print("Best param:", grid_search.best_params_)
    print()

