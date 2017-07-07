import cPickle as pkl
import sys

from data_conversion.databuilder_for_text_files import build_data_from_text_files
from data_conversion.word2vec_util import get_W, load_bin_vec, add_unknown_words

if __name__=="__main__":    
    w2v_file = sys.argv[1]
    folder = sys.argv[2]
    output_pickle = sys.argv[3]
    label_separator = sys.argv[4]

    print "Convert training text files with label separator: " + label_separator,
    print "into to array of words in folder: " + folder
    revs, vocab, label_to_idx, max_words_len, average_words_len = build_data_from_text_files(folder, label_separator, clean_string=True)

    label_to_idx_output = sys.argv[5]
    print "\nDumping label to idx map into:" + label_to_idx_output
    pkl.dump(label_to_idx, open(label_to_idx_output,"wb"))

    print "Training data statistics:"
    print "\tNumber of sentences: " + str(len(revs))
    print "\tVocab size: " + str(len(vocab))
    print "\tMax words len in training data: " + str(max_words_len)
    print "\tAverage words len in training data: " + str(average_words_len)

    # max_vocab_size = int(sys.argv[8])
    # print "\nChoosing top-" + sys.argv[8] + "-most common words",
    # top_vocab = vocab.most_common(max_vocab_size)
    # vocab = dict(top_vocab)
    # print len(vocab)
    # print "No 1", top_vocab[0][0], top_vocab[0][1]
    # print "No", max_vocab_size, top_vocab[-1][0], top_vocab[-1][1]
    
    print "\nLoading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "\tNum words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)

    print "\nDumping training data and word vectors into:" + output_pickle
    pkl.dump(revs, open(output_pickle, "wb"))

    word_vectors_matrix_output = sys.argv[6]
    print "Dumping word vectors matrix into: " + word_vectors_matrix_output
    pkl.dump(W, open(word_vectors_matrix_output,"wb"))

    word_idx_map_output = sys.argv[7]
    print "Dumping word idx map into: " + word_idx_map_output + '\n'
    pkl.dump(word_idx_map, open(word_idx_map_output,"wb"))
