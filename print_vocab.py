from collections import Counter
import cPickle as pkl
from os import listdir
from os.path import isfile, join
import sys

from data_conversion.common_preprocess import  clean_str, normalize_to_unicode, tokenize
vocab_input =''
vocab = pkl.load(open(vocab_input,"rb"))

print 'Vocab size =', len(set(vocab))
f = open('vocab.txt', 'w')
for e in vocab:
    f.write(e)
f.close()