import numpy as np
from hashlib import sha224, md5, sha1, sha256
from sklearn import svm, cross_validation
import random
from os import path
import time
import nltk
import string
import matplotlib.pyplot as plt

def cv(file_name, k=5, save=False):
    items = []
    with file(file_name) as f:
      for line in f:
          line = line.strip()
          items.append(line)

    random.shuffle(items)
    size = len(items)
    test_size = size / k
    extension = file_name.split(".")[1]
    directory = path.dirname(file_name)

    for i in xrange(0, size - test_size + 1, test_size):
        train = items[:i] + items[i + test_size:]
        test = items[i:i + test_size]

        if save:
            with file(directory + "/train_" + str(i) + "." + extension, 'w') as o:
                for line in train:
                    o.write(line + '\n')

            with file(directory + "/test_" + str(i) + "." + extension, 'w') as o:
                for line in test:
                    o.write(line + '\n')

        yield train, test

def get_data(file_name, test=False):
    phrases = {}
    sentiments = {}
    sentences = []
    current_sentence = 0
    #passed a file name
    if type(file_name) == str:
        with file(file_name) as f:
            for line in f:
                #only looking at phrase lines
                if line[0].isdigit():
                    line = line.strip()
                    s = line.split('\t')
                    phrase_id = int(s[0]) - 1
                    sent_id = s[1]
                    sentiment = -1
                    if not test:
                        sentiment = s[-1]

                    if not test:
                        phrase = "\t".join(s[2:-1]).strip().lower()
                    else:
                        phrase = "\t".join(s[2:]).strip().lower()
                    if current_sentence != sent_id:
                        current_sentence = sent_id
                        sentences.append(phrase)

                    phrases[phrase] = phrase_id

                    if not test:
                        sentiments[phrase_id] = int(sentiment)
    #passed the contents of a file
    else:
        for line in file_name:
            #only looking at phrase lines
            if line[0].isdigit():
                line = line.strip()
                s = line.split('\t')
                phrase_id = int(s[0]) - 1
                sent_id = s[1]
                sentiment = -1
                if not test:
                    sentiment = s[-1]

                if not test:
                    phrase = "\t".join(s[2:-1]).strip().lower()
                else:
                    phrase = "\t".join(s[2:]).strip().lower()
                if current_sentence != sent_id:
                    current_sentence = sent_id
                    sentences.append(phrase)

                phrases[phrase] = phrase_id

                if not test:
                    sentiments[phrase_id] = int(sentiment)

    return phrases, sentiments, sentences

def raw_data_analysis(phrases, sentiments, sentences):
    #counts
    phrase_histogram = [0] * 5
    sentence_histogram = [0] * 5
    sentences = set(sentences)

    for phrase, phrase_id in phrases.iteritems():
        if phrase in sentences:
            sentence_histogram[sentiments[phrase_id]] += 1
        else:
            phrase_histogram[sentiments[phrase_id]] += 1

    total_phrase = 0.0
    total_sent = 0.0
    for p, s in zip(phrase_histogram, sentence_histogram):
        total_phrase += p
        total_sent += s

    for i in xrange(5):
        phrase_histogram[i] = phrase_histogram[i] / total_phrase
        sentence_histogram[i] = sentence_histogram[i] / total_sent

    print "Phrase Histogram:"
    print phrase_histogram
    plt.bar(range(0,5), phrase_histogram)
    plt.title("Phrase Histogram")
    plt.show()
    print "Sentence Histogram:"
    print sentence_histogram
    plt.bar(range(0,5), sentence_histogram)
    plt.title("Sentence Histogram")
    plt.show()

    #word frequencies
    print "Top words"
    frequencies = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}}
    ignore = ["film", "movie", "--", "...", "'s",'``',"''", '-lrb-', '-rrb-']
    stop = set(nltk.corpus.stopwords.words('english') + list(string.punctuation) + ignore)

    #tokenize
    for sent in sentences:
        sentiment = sentiments[phrases[sent]]

        tokens = [i for i in nltk.word_tokenize(sent.lower()) if i not in stop]
        for t in tokens:
            count = frequencies[sentiment].get(t, 0)
            frequencies[sentiment][t] = count + 1

    #get words in common
    frequencies_as_sets = [None] * 5
    for i in xrange(5):
        frequencies_as_sets[i] = set(frequencies[i].keys())

    common = set()
    for i in xrange(5):
        for j in xrange(i + 1, 5):
            common = common | (frequencies_as_sets[i] & frequencies_as_sets[j])

    for label, freq in frequencies.iteritems():
        counts = []
        unique = []
        for token, c in freq.iteritems():
            counts.append((token, c))
            if token not in common:
                unique.append((token, c))

        counts = sorted(counts, key= lambda x: x[1], reverse=True)
        unique = sorted(unique, key= lambda x: x[1], reverse=True)
        print label
        print counts[:10]
        print "Unique"
        print unique[:10]

def get_sublists(sentence, n=None):
    sentence_list = sentence.split(" ")
    result = []
    for i in xrange(len(sentence_list) + 1):
        for j in xrange(i + 1, len(sentence_list) + 1):
            if n == None or (j - i) <= n:
                result.append((" ".join(sentence_list[i:j]), j - i))

    return result

#BASELINE
#PROBLEM: one very positive or one very negative word can dominate even if the sentence is not positive or negative
def histogram_of_scores(sentence, phrases, sentiments, ignore_two=False, tag_pos=False, only_adj=False, weighted=False):
    scores = np.zeros(5)

    if tag_pos or only_adj:
        tagged_sent = nltk.pos_tag(sentence.lower().split())
        useful_tags = set(['JJ'])
        adjectives = set([word for word, tag in tagged_sent if tag in useful_tags])
        if only_adj:
            grams = [ (word, 1) for word in adjectives]
        else:
            sub_phrases = get_sublists(sentence.lower())
            grams = []
            for word, weight in sub_phrases:
                not_found = True
                for adj in adjectives:
                    if adj in word and not_found:
                        grams.append((word, weight))
                        not_found = False
    else:
        grams = get_sublists(sentence.lower())

    for word, weight in grams:
        if word in phrases:
            phrase_id = phrases[word]
            s = sentiments[phrase_id]
            if weighted:
                scores[s] += weight
            else:
                scores[s] += 1

    #normalization
    total = 0
    for s in scores:
        total += s

    if total != 0:
        for i in xrange(len(scores)):
            scores[i] = scores[i] / total

    best = 0
    result = -1
    if ignore_two:
        for i in xrange(len(scores)):
            if scores[i] > best and i != 2:
                result = i
                best = scores[i]
    else:
        for i in xrange(len(scores)):
            if scores[i] > best:
                result = i
                best = scores[i]

    #if no words from the phrase are in our test set
    if result == -1:
        result = 2

    return result, scores

#Returns a feature vector
def bow_variation(sentence, size=5001):
    feat = np.zeros(size)

    #hash all unigrams and bigrams
    #weight it by n
    ngrams = get_sublists(sentence.lower(), n = 2)
    for phrase, n in ngrams:
        bucket1 = int(md5(phrase).hexdigest(), 16) % size
        sign1 = int(sha224(phrase).hexdigest(), 16) % 2
        bucket2 = int(sha1(phrase).hexdigest(), 16) % size
        sign2 = int(sha256(phrase).hexdigest(), 16) % 2

        if sign1 == 0:
            feat[bucket1] -= n
        else:
            feat[bucket1] += n

        if sign2 == 0:
            feat[bucket2] -= n
        else:
            feat[bucket2] += n

    #get length
    total = 0.0
    for f in feat:
        total += f

    #normalize
    if total != 0:
        for i in xrange(len(feat)):
            feat[i] = feat[i] / total

    return feat

def test_svm_bow_var(file_name, k=5):
    start_overall = time.clock()
    print "reading data"
    phrases, sentiments, _ = get_data(file_name)
    features = []
    sentiment_array = []

    start = time.clock()
    print "Creating features..."
    for p, p_id in phrases.iteritems():
        feat = bow_variation(p)
        features.append(feat)
        sentiment_array.append(sentiments[p_id])

    X = np.array(features)
    Y = np.array(sentiment_array)
    end = time.clock()
    print "Duration %d" % (end - start)
    
    kf = cross_validation.KFold(len(features), n_folds=k)
    accuracy = 0.0
    print "\nDoing cross validation..."
    for train, test in kf:
        start = time.clock()
        X_train, X_test = X[train], X[test]
        y_train, y_test = Y[train], Y[test]
        print "Training"
        #clf = svm.SVC()
        clf = svm.LinearSVC()
        clf.fit(X_train, y_train)
        print "Testing"
        acc = clf.score(X_test, y_test)
        accuracy += acc
        print acc
        end = time.clock()
        print "Duration %d" % (end - start)

    accuracy = accuracy / k

    end_overall = time.clock()

    print "Total Duration %d" % (end_overall - start_overall)

    return accuracy

def test_svm_hos(file_name, k=5, ignore_two=False, tag_pos=False, only_adj=False, weighted=False):
    accuracy = 0.0
    for train, test in cv(file_name, k=k):
        print "reading in data"
        trainPhrases, trainSentiments, _ = get_data(train)
        testPhrases, testSentiments, _ = get_data(test)

        feat_train = []
        sent_train = []
        feat_test = []
        sent_test = []

        print "creating train features"
        #create train features
        for p, p_id in trainPhrases.iteritems():
            s = trainSentiments[p_id]
            _, feat = histogram_of_scores(p, trainPhrases, trainSentiments, ignore_two=ignore_two, tag_pos=tag_pos, only_adj=only_adj, weighted=weighted)
            feat_train.append(feat)
            sent_train.append(trainSentiments[p_id])

        print "creating test features"
        #create test features
        for p, p_id in testPhrases.iteritems():
            s = testSentiments[p_id]
            _, feat = histogram_of_scores(p, trainPhrases, trainSentiments, ignore_two=ignore_two, tag_pos=tag_pos, only_adj=only_adj, weighted=weighted)
            feat_test.append(feat)
            sent_test.append(testSentiments[p_id])
            
        X_train = np.array(feat_train)
        y_train = np.array(sent_train)
        X_test = np.array(feat_test)
        y_test = np.array(sent_test)

        print "training SVM"
        #create SVM
        #clf = svm.SVC()
        clf = svm.LinearSVC()
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        accuracy += acc
        print acc

    accuracy = accuracy / k

    return accuracy

def test_hos(file_name, k=5, ignore_two=False, tag_pos=False, only_adj=False, weighted=False):
    #train_set = ["train_0.tsv", "train_39015.tsv", "train_78030.tsv", "train_117045.tsv"]
    #test_set = ["test_0.tsv", "test_39015.tsv", "test_78030.tsv", "test_117045.tsv"]

    accuracy = 0.0
    for train, test in cv(file_name, k=k): #zip(train_set, test_set):
        trainPhrases, trainSentiments, _ = get_data(train)
        testPhrases, testSentiments, _ = get_data(test)

        acc = 0.0
        for p, p_id in testPhrases.iteritems():
            s = testSentiments[p_id]
            guess,_ = histogram_of_scores(p, trainPhrases, trainSentiments, ignore_two=ignore_two, tag_pos=tag_pos, only_adj=only_adj, weighted=weighted)
            if s == guess:
                acc += 1

        acc = acc / len(testSentiments)
        accuracy += acc
        print acc

    accuracy = accuracy / k

    return accuracy

if __name__=="__main__":
    acc = test_svm_bow_var("data/train.tsv")
    print acc
