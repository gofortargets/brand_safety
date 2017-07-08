import cPickle as pkl
from os import listdir
from os.path import isfile, join
import sys
from knx.text.preprocess_text import NormalizationText
import re
import nltk

source_dir = sys.argv[1]
output_dir = sys.argv[2]
file_list = [ f for f in listdir(source_dir) if isfile(join(source_dir,f)) ]

print 'Clean data ... at', source_dir

def remove_duplicate(raw):
    def f7(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    res = nltk.sent_tokenize(raw)
    res = [e.strip() for e in res]
    res = f7(res)
    return ' '.join(res)

for true_filename in file_list:
    file_path = join(source_dir, true_filename)
    f_reader = open(file_path, "r")
    raw = f_reader.read()
    content = NormalizationText.preprocess(raw)
    content = remove_duplicate(content)

    output_file_path = join(output_dir, true_filename)
    f_writer = open(output_file_path,"w")
    f_writer.write(content)
    f_writer.close()
