from collections import defaultdict, Counter
import joblib
import json
import numpy as np
import os
import pathlib

cwd = pathlib.Path(os.getcwd())

def preprocess(data):
    """
    Perform the following commands to set up the WordNetLemmatizer
    import nltk
    nltk.download("all")

    This function preprocess the data using the following steps:
    - lowercase
    - remove hyphens
    - remove numbers, punctuations and special characters
    - remove unit of measurements
    - lemmatize words
    """
    from nltk.stem import WordNetLemmatizer
    import re
    from unidecode import unidecode
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
    Given two class clusters A, B.
    This function will filter the data to only contain recipes from the two classes clusters

    Recipes from class cluster A will be labelled 0 and B, 1.
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
    Given a model, it will separate the data into two subsets (A, B)

    Subset A correspond to the data predicted with label 0 by the model
    Subset B correspond to the data predicted with label 1 by the model

    Children contains the names of the subsets

    return a dictionary mapping the children name to subset
    '''
    from scipy.sparse import vstack

    data = vstack(data)
    preds = model.predict(data)

    zero = np.where(preds == 0)
    one = np.where(preds == 1)

    return {children[0]: (data[zero], labels[zero]),
            children[1]: (data[one], labels[one])}


# Train, test functions
def loadTrainData():
    data = json.load(open(str(cwd/"data"/"train.json")))
    return np.array(data["data"]), np.array(data["label"])

def loadTestData():
    data = json.load(open(str(cwd/"data"/"test.json")))
    return np.array(data["data"]), np.array(data["label"])


# Manual five fold validation functions
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
    """"
    Return the preprocessed training data (eliminate installation requirements of nltk and unidecode)
    """
    data = json.load(open(str(cwd / "preprocessed" / "train.json")))
    return np.array(data["data"]), np.array(data["label"])

def loadOfficialTestData():
    data = json.load(open(str(cwd / "original" / "test.json")))
    ingredients = np.array([d['ingredients'] for d in data])
    id = np.array([d['id'] for d in data])
    return id, ingredients

def loadPreprocessTest():
    """"
    Return the preprocessed test data (eliminate installation requirements of nltk and unidecode)
    """
    data = json.load(open(str(cwd / "preprocessed" / "test.json")))
    return np.array(data["id"]), np.array(data["data"])

def saveModel(model, name):
    joblib.dump(model, str(cwd/"models"/name))

def loadModel(name, folder):
    """
    Simply returns model

    model can be encoded in the form of a dictionary with keys {preprocess, model, labelencoder}
    """
    data = joblib.load(str(cwd / folder / name))
    return data

def loadPreprocessModel(modelclass, ckpt, size):
    """
    Used for triplet loss encoding and autoencoder
    """
    import torch
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = modelclass(size)

    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    model.to(device)

    return model

def PMI(data, labels):
    """
    Data: list of documents (each document is a list of ingredients)
    
    Return the Pointwise Mutual Information sorted by class 
    """
    from sklearn.feature_extraction.text import CountVectorizer
    
    data = [" ".join(t) for t in data]
    vectorizer = CountVectorizer()

    x = vectorizer.fit_transform(data)
    words = np.array(sum(x).todense())[0]

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

    return PMI


def filterVocab(data, labels, mag):
    """
    Words with PMI close to zero (within distance of mag to zero) in each classes
    Words which are eliminated for all classes forms the stopwords

    return stopwords
    """
    from sklearn.feature_extraction.text import CountVectorizer

    pmi = PMI(data, labels)

    data = [" ".join(t) for t in data]
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(data)
    vocab = [x[0] for x in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1])]

    filtered = defaultdict(list)
    for k, p in pmi.items():
        filtered[k] = list(filter(lambda x: abs(x[1]) > mag, p.items()))

    remaining = set([w for k, v in filtered.items() for w, s in v])
    remaining_words = [vocab[r] for r in remaining]
    stopwords = list(filter(lambda x: x not in remaining_words, vocab))

    return stopwords


def getPMIScore(word, pmi):
    score = np.zeros(len(pmi.items()))
    for i, (k, v) in enumerate(pmi.items()):
        if word in v:
            score[i] = v[word]
    return score