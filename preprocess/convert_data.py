"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""

import cPickle as pkl
import sys
import warnings

from data_conversion.common_processing import make_vectors_data

warnings.filterwarnings("ignore")

if __name__=="__main__":
    print("Loading data ...")
    mr_p_file = sys.argv[1]
    revs = pkl.load(open(mr_p_file,"rb"))

    word_idx_map_input = sys.argv[2]
    word_idx_map = pkl.load(open(word_idx_map_input,"rb"))

    X_output = sys.argv[3]
    Y_output = sys.argv[4]

    print "Converting text data to matrix ..."
    X, Y =  make_vectors_data(revs, word_idx_map)

    print "Dumping X into:" + X_output + "..."
    pkl.dump(X, open(X_output, "wb"))

    print "Dumping Y into:" + Y_output + "..."
    pkl.dump(Y, open(Y_output, "wb"))    
