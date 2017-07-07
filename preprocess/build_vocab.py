#I'd suggest to reduce the size of the vocabulary by selecting top-k most frequent words / top-k by TfIdf. ' \
# 'Most words occur only once or few times in your dataset.

from collections import Counter
import cPickle as pkl
from os import listdir
from os.path import isfile, join
import sys

from data_conversion.common_preprocess import  clean_str, normalize_to_unicode, tokenize

traindir = sys.argv[1] 
vocab_output = sys.argv[2]
train_file_list = [ f for f in listdir(traindir) if isfile(join(traindir,f)) ]

vocab = Counter()
max_words_len, accumulated_words_len = 0, 0

print "Building vocab from", traindir
for true_filename in train_file_list:
    file_path = join(traindir, true_filename)
    
    f_reader = open(file_path, "r")

    for text in f_reader:
        text = normalize_to_unicode(text) #perform normalization
        text = clean_str(text)
        #add text -> vocab
        words = tokenize(text)
        vocab.update(words)

    f_reader.close()

print "Vocab size:", len(vocab)
print vocab.most_common(20)
print "Dumping vocab files into", vocab_output, '\n'
pkl.dump(vocab,open(vocab_output,'wb'))