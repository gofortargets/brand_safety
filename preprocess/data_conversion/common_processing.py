
def get_idx_from_sent_no_padding(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])

    return x

def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_vectors_data(revs , word_idx_map):
    X = []
    Y = []

    for rev in revs:
        sent_Keras = get_idx_from_sent_no_padding(rev["text"], word_idx_map)
        X.append(sent_Keras)
        Y.append(rev["y"])

    return X, Y

def make_vectors_data_no_label(revs , word_idx_map):
    X = []
    for rev in revs:
        sent_Keras = get_idx_from_sent_no_padding(rev["text"], word_idx_map)
        X.append(sent_Keras)

    return X