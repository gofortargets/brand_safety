from collections import defaultdict, Counter
from os import listdir
from os.path import isfile, join

import numpy as np

from data_conversion.common_preprocess import  clean_str, normalize_to_unicode

def build_data_from_text_files(traindir, label_separator, clean_string=True):
    #build data from text files
    train_file_list = [ f for f in listdir(traindir) if isfile(join(traindir,f)) ]

    revs = []
    # vocab = defaultdict(float)
    vocab = Counter()
    next_label_id = 0
    label_to_idx = defaultdict(float)
    max_words_len, accumulated_words_len = 0, 0
    count = 0
    for true_filename in train_file_list:
        #process label (for y field)
        idx_sep = true_filename.find(label_separator)
        if idx_sep == -1:
            continue
        label = true_filename[0:idx_sep]

        #process content (for text field)
        file_path = join(traindir, true_filename)
        f_reader = open(file_path, "r")
        text = f_reader.read() #read text
        text = normalize_to_unicode(text) #perform normalization
        f_reader.close()
        if clean_string:
            orig_rev = clean_str(text)
        else:
            orig_rev = text.lower()

        #process vocab
        vocab.update(set(orig_rev.split()))

        #build label to idx
        label_id = -1
        if label in label_to_idx.keys():
            label_id = label_to_idx[label]
        else:
            label_to_idx[label] = next_label_id
            label_id = next_label_id
            next_label_id = next_label_id + 1

        #build a dictionary map of label
        num_words_count = len(orig_rev.split())
        datum = {   "y": label_id,
                    "text": orig_rev,
                    "num_words": num_words_count
                } 

        if num_words_count > max_words_len:
            max_words_len = num_words_count

        accumulated_words_len = accumulated_words_len + num_words_count
        count = count + 1
        revs.append(datum)

    average_words_len = accumulated_words_len*1.0/count
    return revs, vocab, label_to_idx, max_words_len, average_words_len

