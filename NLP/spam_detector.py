from nltk.corpus import movie_reviews
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
import nltk
import random
import pickle
import os.path

naivebayes = "/Users/daniellungu/Documents/Workspace/Natural-Language-Processing/naivebayes.pickel"

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

words_feature = list(all_words.keys())[:3000]


def find_feature(document):
    words = set(document)
    feature = {}
    for w in words_feature:
        feature[w] = (w is words)

    return feature


features = [(find_feature(rev), category) for (rev, category) in documents]

testing_set = features[1900:]
training_set = features[:1900]

if not os.path.isfile(naivebayes):
    classifier = nltk.NaiveBayesClassifier.train(training_set)

    save_classifier = open(naivebayes, "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()
else:
    classifier_f = open(naivebayes, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()

print("Original Naive Bayes Classifier accuracy precent:", (nltk.classify.accuracy(classifier, testing_set) * 100))

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("Multinomial Naive Bayes Classifier accuracy precent:", (nltk.classify.accuracy(classifier, testing_set) * 100))
