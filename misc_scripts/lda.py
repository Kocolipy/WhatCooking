import gensim
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from collections import Counter, defaultdict
from sklearn.svm import SVC
import numpy as np
import utils

def convertToArray(data, size):
    arr = np.zeros(size)
    for (i,v) in data:
        arr[i] = v
    return arr


def clean(data):
    data = [" ".join(t) for t in data]
    vectorizer = CountVectorizer()

    x = vectorizer.fit_transform(data)
    vocab = [x[0] for x in sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1])]
    cleaned = [[vocab[w] for w in sample.indices] for sample in x]
    return cleaned

if __name__ == '__main__':
    feature_size = 2

    train, train_label, val, val_label = utils.loadFoldData(1)

    # train = clean(train)

    grouped_docs = defaultdict(list)
    for i, c in enumerate(train_label):
        grouped_docs[c] += train[i]

    # Label Encoding - Target
    print("Label Encode the Target Variable ... ")
    lb = LabelEncoder()
    oh_train_label = lb.fit_transform(train_label)

    dictionary = gensim.corpora.Dictionary(grouped_docs.values())
    # dictionary.filter_extremes(no_below=1, no_above=0.5)

    bow_corpus = {}
    for k, doc in grouped_docs.items():
        bow_corpus[k] = dictionary.doc2bow(doc)
    # train_ft = [convertToArray(dictionary.doc2bow(doc), len(dictionary.keys())) for doc in processed_docs]

    lda_model = gensim.models.LdaMulticore(bow_corpus.values(), num_topics=feature_size, id2word=dictionary, passes=10, workers=2)

    for k , bc in bow_corpus.items():
        print(k, lda_model[bc])
    # train_ft = [convertToArray(lda_model[doc], feature_size) for doc in bow_corpus]

    print()

    # print("Train the model ... ")
    # classifier = SVC(C=10,  # penalty parameter
    #                  kernel='sigmoid',  # kernel type, rbf working fine here
    #                  gamma=0.1,
    #                  coef0=1,  # kernel coefficient
    #                  tol=0.001,  # stopping criterion tolerance
    #                  )
    # model = OneVsRestClassifier(classifier, n_jobs=4)
    #
    # model.fit(train_ft, oh_train_label)

    print("Predict on validation data ... ")
    val_im = [" ".join(t).lower().split(" ") for t in val]
    val_im = [dictionary.doc2bow(doc) for doc in val_im]
    # val_ft = [convertToArray(dictionary.doc2bow(doc), len(dictionary.keys())) for doc in val_im]
    val_ft = [convertToArray(lda_model[doc], feature_size) for doc in val_im]

    val_out = model.predict(val_ft)
    val_pred = lb.inverse_transform(val_out)

    acc = np.sum(val_label == val_pred) / len(val_pred)
    print(acc)