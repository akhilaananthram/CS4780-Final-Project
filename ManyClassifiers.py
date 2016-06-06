#from stat_parser import Parser, display_tree
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn import svm
from scipy.sparse import coo_matrix
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import Perceptron
from sklearn import linear_model, dummy
import random
import math
import argparse


# Things to Do:
# 1. Non-Vectorial Parse Trees with Tree Kernels
# 2. Implement tf*idf
# 3. Potentially add 2-grams to feature vector (this might be too much data to train on however)
# 4. Hyper-Parameter Optimization using Grid Search
# 5. Submit to Kaggle to determine accuracy on test set

# Classification Algorithms:
# 1. SVM with Linear Kernel
# 2. Multinomial Naive Bayes
# 3. Stochastic Gradient Descent
# 4. Nearest Centroid
# 5. Perceptron
# 6. Ridge Classification
# Note: Random Forest and Decision Tree classifiers can't be implemented because they don't currently support sparse matrices
# Note: SVM with non-linear Kernels can't be implemented because they take too long to train and consequentely perform cross-validation. Also, non-linear models will tend to overfit.

def parse_args():
    parser = argparse.ArgumentParser(description="Train Sentiment Analysis Classifiers")
    parser.add_argument("--train", dest="train", type=str, help="Path to training data. REQUIRED", required=True)
    parser.add_argument("--test", dest="test", type=str, help="Path to test data. REQUIRED", required=True)

    parser.add_argument("--k", dest="k", type=int, help="number of folds for cross validation", default=5)
    parser.add_argument("--stop", dest="stop", action="store_true")
    parser.add_argument("--stem", dest="stem", action="store_true")
    parser.add_argument("--classifier", choices=["linear", "ovo", "NB", "SGD", "centroid", "perceptron", "ridge", "levels"], default="linear")

    args = parser.parse_args()
    return args

class LevelClassifier(dummy.DummyClassifier):
    def __init__(self):
        dummy.DummyClassifier.__init__(self)
        self.levels = []

    def add_classifier(self, classifier, level, label=""):
        if len(self.levels) < level:
            self.levels.append({})

        l = self.levels[level - 1][label] = classifier

    def predict(self, X):
        if self.levels == []:
            return None

        label = ""
        for level in self.levels:
            if type(label) != str:
                label = label[0]
            classifier = level.get(label, None)
            if classifier is None:
                return label

            label = classifier.predict(X)

        if type(label) != str:
            label = label[0]

        return label

def kFoldCrossValidation(all_training_data, trainingphrases_to_sentiment, vocabulary, k = 5, classifier = "linear"):
    random.shuffle(all_training_data)
    number_of_examples = len(all_training_data)
    size_of_test_set = number_of_examples / k
    accuracy = 0.0


    for i in xrange(0, number_of_examples - size_of_test_set + 1, size_of_test_set):

        trainingdata = all_training_data[:i] + all_training_data[i + size_of_test_set:]
        testdata = all_training_data[i:i + size_of_test_set]


        print "Building Feature Vectors..."
        X_train, Y_train = getXandY(trainingdata, vocabulary, trainingphrases_to_sentiment, tfidf = False, L1normalization = False, L2normalization = True)
        X_test, Y_test = getXandY(testdata, vocabulary, trainingphrases_to_sentiment, tfidf = False, L1normalization = False, L2normalization = True)


        print "Training..."
        if classifier == "linear":
            model = svm.LinearSVC()
            model.fit(X_train, Y_train)
        elif classifier == "ovo":
            model = svm.SVC()
            model.fit(X_train, Y_train)
        elif classifier == "NB":
            model = MultinomialNB()
            model.fit(X_train, Y_train)
        elif classifier == "SGD":
            model = linear_model.SGDClassifier()
            model.fit(X_train, Y_train)
        elif classifier == "centroid":
            model = NearestCentroid()
            model.fit(X_train, Y_train)
        elif classifier == "perceptron":
            model = Perceptron()
            model.fit(X_train, Y_train)
        elif classifier == "ridge":
            model = RidgeClassifier()
            model.fit(X_train, Y_train)
        elif classifier == "levels":
            #coarse classifier
            print "Training top level"
            Y_train_coarse = []
            for y in Y_train:
                if int(y) < 2:
                    Y_train_coarse.append("low")
                elif int(y) == 2:
                    Y_train_coarse.append("2")
                else:
                    Y_train_coarse.append("high")

            model = LevelClassifier()
            top = svm.LinearSVC()
            top.fit(X_train, Y_train_coarse)
            model.add_classifier(top, 1, "")

            print "Training Low"
            X_low, Y_low =  getXandY(trainingdata, vocabulary, trainingphrases_to_sentiment, tfidf = False, L1normalization = False, L2normalization = True, separate="low")
            low = svm.LinearSVC()
            low.fit(X_low, Y_low)
            model.add_classifier(low, 2, "low")

            print "Training High"
            X_high, Y_high =  getXandY(trainingdata, vocabulary, trainingphrases_to_sentiment, tfidf = False, L1normalization = False, L2normalization = True, separate="high")
            high = svm.LinearSVC()
            high.fit(X_high, Y_high)
            model.add_classifier(high, 2, "high")

        print "Testing..."
        accuracy += model.score(X_test, Y_test)

    return accuracy/k

def generate_vocabulary(all_sentences):
    vocabulary = {}
    word_counter = 0
    for sentence in all_sentences:

        word_list = sentence.split(' ')

        for word in word_list:
            if word not in vocabulary:
                vocabulary[word] = word_counter
                word_counter += 1

    return vocabulary


def get_idf_values(training_sentences):

    word_to_number_of_sentences_that_contain_it = {}
    word_to_idf_value = {}

    for sentence in training_sentences:
        word_list = sentence.split(' ')
        for word in list(set(word_list)):

            if word not in word_to_number_of_sentences_that_contain_it:
                word_to_number_of_sentences_that_contain_it[word] = 1
            else:
                word_to_number_of_sentences_that_contain_it[word] += 1

    for word, frequency in word_to_number_of_sentences_that_contain_it.iteritems():
        word_to_idf_value[word] = math.log(len(training_sentences)/frequency)

    return word_to_idf_value

# Three possible sparse feature vectors can be created
# 1. Binary (either it contains the word or not)
# 2. tf weights
# 3. tf*idf weights

def getXandY(phrases, vocabulary, phrases_to_sentiment, tf = True, tfidf = False, L1normalization = False, L2normalization = False, separate=None):

    row = []
    col = []
    data = []
    Y = []
    current_row = 0
    valid_phrases = 0

    for phrase in phrases:
        s = int(phrases_to_sentiment[phrase])
        if (separate is None) or (separate == "low" and s < 2) or (separate == "high" and s > 2):
            valid_phrases += 1

            #determine the frequencies of each word within this phrase

            phrase_frequency_vector = {}
            word_list = phrase.split(' ')
            L1total = 0.0
            for word in word_list:
                if word not in phrase_frequency_vector:
                    phrase_frequency_vector[word] = 1
                    L1total += 1.0
                else:
                    if tf:
                        phrase_frequency_vector[word] += 1
                        L1total += 1.0

            #normalize the feature vector if option was enabled

            if L1normalization:
                for word, frequency in phrase_frequency_vector.iteritems():
                    phrase_frequency_vector[word] = frequency/L1total

            if L2normalization:
                L2total = 0.0
                for word, frequency in phrase_frequency_vector.iteritems():
                    L2total += math.pow(frequency, 2)
                L2total = math.pow(L2total, 0.5)
                for word, frequency in phrase_frequency_vector.iteritems():
                    phrase_frequency_vector[word] = frequency/L2total

            #construct the feature vector in sparse form

            for word in word_list:

                if tf and tfidf:
                    data.append(phrase_frequency_vector[word]*idf_values[word])
                    row.append(current_row)
                    col.append(vocabulary[word])

                else:
                    data.append(phrase_frequency_vector[word])
                    row.append(current_row)
                    col.append(vocabulary[word])

            #construct the target vector

            Y.append(phrases_to_sentiment[phrase])

            current_row += 1

    X = coo_matrix((np.array(data), (np.array(row), np.array(col))), shape=(valid_phrases, len(vocabulary)))

    print valid_phrases
    return X, np.array(Y)


def construct_explicit_feature_vector(phrase, vocabulary, tfidf = False):
    feature_vector = np.zeros(len(vocabulary))

    word_list = phrase.split(' ')
    for word in word_list:
        feature_vector[vocabulary[word]] += 1

    if tfidf:
        for word in word_list:
            word_number = vocabulary[word]
            feature_vector[word_number] = feature_vector[word_number] * idf_values[word]

    return feature_vector

def stop(sentence):

    stoppingwords = set(stopwords.words('english'))
    word_list = sentence.split(' ')
    words_after_stopping = [w for w in word_list if not w in stoppingwords]

    if '.' in words_after_stopping:
        words_after_stopping[:] = (value for value in words_after_stopping if value != '.')

    if ',' in words_after_stopping:
        words_after_stopping[:] = (value for value in words_after_stopping if value != ',')

    return ' '.join(words_after_stopping)

def stem(sentence):

    stemmer = SnowballStemmer("english")
    word_list = sentence.split(' ')
    words_after_stemming = []
    for word in word_list:
        words_after_stemming.append(stemmer.stem(word))

    if '.' in words_after_stemming:
        words_after_stemming[:] = (value for value in words_after_stemming if value != '.')

    if ',' in words_after_stemming:
        words_after_stemming[:] = (value for value in words_after_stemming if value != ',')

    return ' '.join(words_after_stemming)

def extract_data(file_name, istestData=False, stopping= False, stemming = False):
    sentences = []
    phrases = []
    phrases_to_phrase_id = {}
    phrases_to_sentence_id = {}
    phrases_to_sentiment = {}
    current_sentence = 0

    with file(file_name) as f:
        for line in f:
            if line[0].isdigit():
                line = line.strip()
                s = line.split('\t')

                #extract phraseid and phrase, performing any stopping and stemming if necessary
                phrase_id = int(s[0])
                if not istestData:
                    phrase = "\t".join(s[2:-1]).strip().lower()
                    if stopping:
                        phrase = stop(phrase)
                    if stemming:
                        phrase = stem(phrase)
                else:
                    phrase = "\t".join(s[2:]).strip().lower()
                    if stopping:
                        phrase = stop(phrase)
                    if stemming:
                        phrase = stem(phrase)

                #extract sentenceid and sentence
                sentence_id = s[1]
                if current_sentence != sentence_id:
                    current_sentence = sentence_id
                    sentences.append(phrase)

                else:
                    phrases.append(phrase)

                #extract sentiment
                sentiment = -1
                if not istestData:
                    sentiment = s[-1]

                #add all data to dictionaries
                phrases_to_phrase_id[phrase] = phrase_id
                phrases_to_sentence_id[phrase] = sentence_id
                if not istestData:
                    phrases_to_sentiment[phrase] = int(sentiment)

    if not istestData:
        return sentences, phrases, phrases_to_phrase_id, phrases_to_sentence_id, phrases_to_sentiment
    else:
        return sentences, phrases, phrases_to_phrase_id, phrases_to_sentence_id


def get_all_n_grams(sentence, n=None):
    sentence_list = sentence.split(" ")
    result = []
    for i in xrange(len(sentence_list) + 1):
        for j in xrange(i + 1, len(sentence_list) + 1):
            if n is None or (j - i) <= n:
                result.append((" ".join(sentence_list[i:j]), j - i))

    return result

def test(testdata, model, vocabulary, trainingphrases_to_sentiment):
    correct = 0.0
    total = 0.0
    for phrase in testdata:
        feature_vector = construct_explicit_feature_vector(phrase, vocabulary)
        predicated_class = model.predict(feature_vector)
        actual_class = trainingphrases_to_sentiment[phrase]

        if actual_class == predicated_class[0]:
            correct += 1

        total += 1

    return correct/total

if __name__ == "__main__":
    args = parse_args()

    print "Extracting Data..."
    trainingsentences, trainingphrases, trainingphrases_to_phrase_id, trainingphrases_to_sentence_id, trainingphrases_to_sentiment = extract_data(args.train, False, stopping = args.stop, stemming = args.stem)
    #testsentences, testphrases, testphrases_to_phrase_id, testphrases_to_sentence_id = extract_data(args.test, True, stopping = args.stop, stemming = args.stem)

    print "Generating Vocabulary..."
    vocabulary = generate_vocabulary(trainingsentences + testsentences + trainingphrases + testphrases)
    print "Vocabulary Generation Completed with Total Number of Words %d" % len(vocabulary)

    print "Performing k Fold Cross Validation..."
    accuracy = kFoldCrossValidation(trainingsentences+trainingphrases, trainingphrases_to_sentiment, vocabulary, k = args.k, classifier = args.classifier)
    print "k Fold Cross Validation Completed with Accuracy of: " + str(accuracy)
