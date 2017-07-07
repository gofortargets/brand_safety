import cPickle as pkl
import os
import sys

from knx.text.doc_to_feature import DocToFeature, tf_to_tfidf, tf_to_okapi, tf_to_midf, tf_to_rf
from knx.util.logging import Timing

LABEL_SEPARATOR = '#'

def scorer_tf(doc_term_freq, labels=None):
    return doc_term_freq

def scorer_tfidf(doc_term_freq, labels=None):
    (doc_term, idf_diag) = tf_to_tfidf(doc_term_freq, sublinear_tf=True)
    return doc_term

def scorer_okapi(doc_term_freq, labels=None):
    (doc_term, idfs, avg_doc_len) = tf_to_okapi(doc_term_freq)
    return doc_term

def scorer_midf(doc_term_freq, labels=None):
    (doc_term_freq_idf,) = tf_to_midf(doc_term_freq)
    return doc_term_freq_idf

def scorer_rf(doc_term_freq, labels=None):
    (doc_term, rf_vectors) = tf_to_rf(doc_term_freq, labels=labels)
    return doc_term

scorers = {
    'tf': scorer_tf,
    'tfidf': scorer_tfidf,
    'okapi': scorer_okapi,
    'midf': scorer_midf,
    'rf': scorer_rf
}

def get_scorer(scorer_name):
    return scorers[scorer_name]

def train_on_directory(scorer_name, traindir, lowercase=True, word_normalization="stem"):
    with Timing('Processing training files in the folder %s...' % traindir):
        dtf = DocToFeature(lowercase=lowercase, word_normalization=word_normalization)
        if type(traindir) == list:
            file_list = []
            for dirname in traindir:
                if not dirname.endswith('/'):
                    dirname += '/'
                file_list.extend((dirname + filename for filename in os.listdir(dirname)
                                  if filename != '.DS_Store'))
            traindir = file_list
        train_doc_term_freq = dtf.doc_to_tf(traindir)
        train_file_list = dtf.filelist
        vocabulary = dtf.vocabulary
        mapping = dtf.mapping
        labels = []
        train_classes = []
        for filename in train_file_list:
            true_filename = filename[filename.rfind('/') + 1:]
            label = true_filename[0:true_filename.find(LABEL_SEPARATOR)]
            labels.append(label)
            if label not in train_classes:
                train_classes.append(label)
        train_classes.sort()

    with Timing('Calculating feature scores using scorer %s...' % scorer_name):
        train_doc_term = get_scorer(scorer_name)(train_doc_term_freq, labels=labels)

    return train_doc_term, labels, train_classes, vocabulary, mapping

def test_on_directory(scorer_name, testdir, vocabulary, lowercase=True, word_normalization="stem"):
    with Timing('Processing test files in the folder %s...' % testdir):
        dtf = DocToFeature(lowercase=lowercase, word_normalization=word_normalization)
        if type(testdir) == list:
            file_list = []
            for dirname in testdir:
                if not dirname.endswith('/'):
                    dirname += '/'
                file_list.extend((dirname + filename for filename in os.listdir(dirname)
                                  if filename != '.DS_Store'))
            testdir = file_list
        test_doc_term_freq = dtf.doc_to_tf(testdir, vocabulary=vocabulary)
        test_file_list = dtf.filelist
        labels = []
        test_classes = []
        true_filename_list = []
        for filename in test_file_list:
            true_filename = filename[filename.rfind('/') + 1:]
            true_filename_list.append(true_filename)
            label = true_filename[0:true_filename.find(LABEL_SEPARATOR)]
            labels.append(label)
            if label not in test_classes:
                test_classes.append(label)
        test_classes.sort()

    if scorer_name == 'midf':
        mesg = 'Calculating feature scores using scorer %s...' % (scorer_name)
    else:
        mesg = ('Calculating feature scores using scorer %s '
                'using estimated collection-specific information...' % (scorer_name)),
    with Timing(mesg):
        test_doc_term = get_scorer(scorer_name)(test_doc_term_freq)

    return test_doc_term, labels, true_filename_list

def main():
    traindir = sys.argv[1]
    testdir = sys.argv[2]
    scorer = sys.argv[3]
    outputdir = sys.argv[4]

    X_train, y_train, train_classes, vocabulary, mapping = train_on_directory(scorer, traindir)
    output_train = os.path.join(outputdir, "train.pkl")
    with open(output_train, "wb") as fp:
        pkl.dump(X_train, fp)
        pkl.dump(y_train, fp)
        pkl.dump(train_classes, fp)
        pkl.dump(vocabulary, fp)
        pkl.dump(mapping, fp)

    X_test, y_test, filename_ls = test_on_directory(scorer, testdir, vocabulary)
    output_train = os.path.join(outputdir, "test.pkl")
    with open(output_train, "wb") as fp:
        pkl.dump(X_test, fp)
        pkl.dump(y_test, fp)
        pkl.dump(filename_ls, fp)

if __name__ == '__main__':
    main()
