import cPickle as pkl
import sys

from data_conversion.databuilder_for_test import build_test_data

if __name__=="__main__":
    test_dir = sys.argv[1]
    label_separator = sys.argv[2]
    label_to_idx = pkl.load(open(sys.argv[3],"rb"))
    test_output = sys.argv[4]

    print "Convert testing text files to array of words in folder:" + test_dir + " ..."
    revs_test, max_words_len_test, average_words_len_test = build_test_data(test_dir, label_to_idx, label_separator, clean_string=True)
    print"Dumping testing data into: " + test_output + " ..."
    pkl.dump(revs_test, open(test_output,"wb"))
    print "Test data statistics:"
    print "\tMax words len for in test data:" + str(max_words_len_test)
    print "\tAverage words len in test data:" + str(average_words_len_test) + '\n'
    