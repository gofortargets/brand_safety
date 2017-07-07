#I'd suggest to reduce the size of the vocabulary by selecting top-k most frequent words / top-k by TfIdf. ' \
# 'Most words occur only once or few times in your dataset.

from collections import Counter
import cPickle as pkl
from os import listdir
from os.path import isfile, join
import sys

from data_conversion.common_preprocess import  clean_str, normalize_to_unicode, tokenize

traindir = sys.argv[1]
output_dir = sys.argv[2]
vocab_input = sys.argv[3]
max_vocab_size = int(sys.argv[4])
train_file_list = [ f for f in listdir(traindir) if isfile(join(traindir,f)) ]

print 'Loading vocab from', vocab_input
vocab = pkl.load(open(vocab_input,"rb"))

print "Max vocab size:", max_vocab_size 
top_words = set()
most_words = vocab.most_common(max_vocab_size) 
for k, v in most_words:
    top_words.add(k)
print most_words[-5:]

print "Generating Vocab-reduced files", output_dir
for true_filename in train_file_list:
    file_path = join(traindir, true_filename)
    f_reader = open(file_path, "r")
    total_text = ""
    for text in f_reader:
        text = normalize_to_unicode(text) #perform normalization
        total_text = total_text + text
    f_reader.close()
    orig_rev = clean_str(total_text.strip())

    #process vocab
    words = tokenize(orig_rev)
    new_words = [word for word in words if word in top_words]

    new_text = " ".join(new_words)
    output_file_path = join(output_dir, true_filename)
    f_writer = open(output_file_path,"w")
    f_writer.write(new_text)
    f_writer.close()

print ""
