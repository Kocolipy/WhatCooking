from collections import defaultdict, Counter
import joblib
import json
import numpy as np
import os
import pathlib
from nltk.stem import WordNetLemmatizer
import re
from unidecode import unidecode

cwd = pathlib.Path(os.getcwd())

def preprocess(data):

    lemmatizer = WordNetLemmatizer()

    data = ' '.join(data)
    data = data.lower()  # Lower - Casing
    data = data.replace('-', ' ')  # Removing Hyphen
    words = []
    for word in data.split():
        word = re.sub("[0-9]", " ", word)  # removing numbers,punctuations and special characters
        word = re.sub((r'\b(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\b'), ' ', word)  # Removing Units
        if len(word) <= 2: continue  # Removing words with less than two characters
        word = unidecode(word)  # Removing accents
        word = lemmatizer.lemmatize(word)  # Lemmatize
        if len(word) > 0: words.append(word)
    return ' '.join(words)


def dataSplit(data, labels, classesA, classesB):
    '''
    return labels of 0 for class A and 1 for class B
    '''
    from numpy.random import permutation

    train_label_dict = {}
    for uni in np.unique(labels):
        train_label_dict[uni] = np.nonzero(labels == uni)[0].tolist()

    new_data = []
    orig_label = []
    new_label = []
    for c in classesA:
        for index in train_label_dict[c]:
            new_data.append(data[index])
            orig_label.append(labels[index])
            new_label.append(0)

    for c in classesB:
        for index in train_label_dict[c]:
            new_data.append(data[index])
            orig_label.append(labels[index])
            new_label.append(1)

    perm = permutation(len(new_data))
    new_data = np.array(new_data)[perm]
    orig_label = np.array(orig_label)[perm]
    new_label = np.array(new_label)[perm]

    return new_data, orig_label, new_label


def adjustTrainingSet(data, labels, children, model):
    '''
    return labels of 0 for class A and 1 for class B
    '''
    from scipy.sparse import vstack

    data = vstack(data)
    preds = model.predict(data)

    zero = np.where(preds == 0)
    one = np.where(preds == 1)

    dct = {
        children[0]: (data[zero], labels[zero]),
        children[1]: (data[one], labels[one])
    }

    return dct

def loadFoldData(fold):
    t, tl = loadFoldTrainData(fold)
    v, vl = loadFoldValData(fold)
    return t, tl, v, vl


def loadFoldTrainData(fold):
    data = json.load(open(str(cwd/"fold_data"/str(fold)/"train.json")))
    # return np.array([[w.lower() for w in t] for t in data["data"]]), np.array(data["label"])
    return np.array([" ".join(t).lower().split(" ") for t in data["data"]]), np.array(data["label"])


def loadFoldValData(fold):
    data = json.load(open(str(cwd/"fold_data"/str(fold)/"val.json")))
    return np.array([" ".join(t).lower().split(" ") for t in data["data"]]), np.array(data["label"])


def loadOriginalTrain():
    data = json.load(open(str(cwd / "original" / "train.json")))
    labels = np.array([d['cuisine'].lower() for d in data])
    ingredients = np.array([d['ingredients'] for d in data])
    return ingredients, labels

def loadPreprocessed():
    data = json.load(open(str(cwd / "preprocessed.json")))
    labels = np.array(data["label"])
    ingredients = np.array(data["data"])
    return ingredients, labels

def loadTrainData():
    data = json.load(open(str(cwd/"data"/"train.json")))
    return np.array(data["data"]), np.array(data["label"])

def loadTestData():
    data = json.load(open(str(cwd/"data"/"test.json")))
    return np.array(data["data"]), np.array(data["label"])


def loadOfficialTestData():
    data = json.load(open(str(cwd / "original" / "test.json")))
    ingredients = np.array([d['ingredients'] for d in data])
    id = np.array([d['id'] for d in data])
    return id, ingredients

def saveModel(model, name):
    joblib.dump(model, str(cwd/"models"/name))

def loadModelSVM(name, folder):
    data = joblib.load(str(cwd / folder / name))
    return data

def loadModel(name):
    data = joblib.load(str(cwd/"models"/name))
    return data["model"], data["labelencoder"]

def loadModelOnly(name):
    data = joblib.load(str(cwd/"models"/name))
    return data

def loadPreprocessModel(modelclass, ckpt, size):
    import torch
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = modelclass(size)

    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    model.to(device)

    return model

def PMI(data, labels, mag):
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    """
    Data: list of documents (each document is a list of ingredients)
    """
    data = [" ".join(t) for t in data]
    vectorizer = CountVectorizer()

    x = vectorizer.fit_transform(data)
    words = np.array(sum(x).todense())[0]
    vocab = [x[0] for x in sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1])]
    print(len(vocab))

    train_label_dict = {}
    for uni in np.unique(labels):
        train_label_dict[uni] = np.nonzero(labels == uni)[0].tolist()

    class_counts = defaultdict()
    for k, v in train_label_dict.items():
        class_counts[k] = len(v)
    total_classes = sum(class_counts.values())

    words_by_class = defaultdict(list)
    for k, v in train_label_dict.items():
        for d in v:
            words_by_class[k] += x[d].indices.tolist()

    # Most common words for each class
    for k, v in words_by_class.items():
        words_by_class[k] = Counter(v)

    # PMI
    PMI = defaultdict(dict)
    for k, v in words_by_class.items():
        for w, c in v.items():
            PMI[k][w] = np.log(c) - np.log(words[w]) - np.log(class_counts[k] / total_classes)

    filtered = defaultdict(list)
    for k, pmi in PMI.items():
        filtered[k] = list(filter(lambda x: abs(x[1]) > mag, pmi.items()))

    remaining = set([w for k, v in filtered.items() for w, s in v])
    print(len(remaining))
    remaining_words = [vocab[r] for r in remaining]
    stopwords = list(filter(lambda x: x not in remaining_words, vocab))

    return stopwords

    # # Retain Top Threshold proportion of words per class based on PMI
    # sortedPMI = defaultdict(list)
    # for k, pmi in PMI.items():
    #     unique_words_class = len(pmi)
    #     for w, score in sorted(pmi.items(), key=lambda x: x[1], reverse=True)[:int(threshold * unique_words_class)]:
    #         sortedPMI[k].append(w)
    #
    allowed_words = set([w for words in filtered.values() for w in words])
    # # Generate filtered documents
    # processed_data = []
    # for i in range(len(data)):
    #     doc = data[i]
    #     new_doc = list(filter(lambda x: x in allowed_words, doc))
    #     processed_data.append(new_doc)
    #     if not new_doc:
    #         print(doc, "will not have any words.")
    #         print("Cuisine Type:", labels[i], i)
    return PMI

def getPMIScore(word, PMI):
    score = np.zeros(len(PMI.items()))
    for i, (k, v) in enumerate(PMI.items()):
        if word in v:
            score[i] = v[word]
    return score

def filterByTotalFreq(data, min_df, max_df):
    words = " ".join([" ".join(t) for t in data]).split(' ')
    words = Counter(words)
    total_words = sum(words.values())
    allowed_words = []
    value = 0
    for w, c in words.most_common():
        value += float(c) / total_words
        if value > max_df and value < min_df:
            allowed_words.append(w)

    processed_data = []
    for i in range(len(data)):
        doc = data[i]
        new_doc = list(filter(lambda x: x in allowed_words, doc))
        processed_data.append(new_doc)
        if not new_doc:
            print(doc, "will not have any words.")
    return processed_data