from os import listdir
from os.path import isfile, join

from data_conversion.common_preprocess import  clean_str, normalize_to_unicode

def build_test_data_no_label_for_one_file(file_path, clean_string=True):
    f_reader = open(file_path, "r")
    text = f_reader.read() #read text
    text = normalize_to_unicode(text) #perform normalization
    f_reader.close()
    if clean_string:
        orig_rev = clean_str(text)
    else:
        orig_rev = text.lower()

    num_words_count = len(orig_rev.split())
    datum = {   "text": orig_rev,
                "num_words": num_words_count}
    return [datum]


def build_test_data_no_label(directory, clean_string=True):
    file_list = [ f for f in listdir(directory) if isfile(join(directory,f)) ]

    revs = []
    max_words_len, accumulated_words_len = 0, 0
    count = 0
    for true_filename in file_list:
        #process content (for text field)
        file_path = join(directory, true_filename)
        f_reader = open(file_path, "r")
        text = f_reader.read() #read text
        text = normalize_to_unicode(text) #perform normalization
        f_reader.close()
        rev = []
        rev.append(text.strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()

        #build a dictionary map of label
        num_words_count = len(orig_rev.split())
        datum = {   "text": orig_rev,
                    "num_words": num_words_count}

        if num_words_count > max_words_len:
            max_words_len = num_words_count

        accumulated_words_len = accumulated_words_len + num_words_count
        count = count + 1
        revs.append(datum)

    average_words_len = accumulated_words_len*1.0/count

    return revs, max_words_len, average_words_len

#for 1 label only <label>#dfdasfd.txt
def build_test_data(directory, label_to_idx, label_separator, clean_string=True):
    file_list = [ f for f in listdir(directory) if isfile(join(directory,f)) ]

    revs = []
    max_words_len, accumulated_words_len = 0, 0
    count = 0
    for true_filename in file_list:
        #process label (for y field)
        idx_sep = true_filename.find(label_separator)
        if idx_sep == -1:
            continue
        label = true_filename[0:idx_sep]

        #process content (for text field)
        file_path = join(directory, true_filename)
        f_reader = open(file_path, "r")
        text = f_reader.read() #read text
        text = normalize_to_unicode(text) #perform normalization
        f_reader.close()
        if clean_string:
            orig_rev = clean_str(text)
        else:
            orig_rev = text.lower()

        #build label to idx
        label_id = label_to_idx[label]

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

    return revs, max_words_len, average_words_len