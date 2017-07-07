import re
from nltk import corpus
from nltk.tokenize import RegexpTokenizer
from HTMLParser import HTMLParser
import string


REPLACEMENT_WORDS = [
    "Editing by", "editing by", "Editing By",
    "Reporting By", "Reporting by", "reporting by",
    "Compiled by", "compiled by", "Compiled By",
    "Writing by", "Writing By", "writing by",
    "U.S.", "U.N.",
    "(RNS)",
    "BERLIN:",  "BIRMINGHAM:",
    "LISBON:",  "MARSEILLE:",
    "PUTRAJAYA:",   "BERNE:",
    "DALLAS:",
    "DHAKA:",
    "SYDNEY:",
    "LARKIN:",
    "INCHEON:",
    "KUCHING:",
    "TOKYO:",
    "MUKAH:",
    "EASTBOURNE:",
    "MUGELLO:",
    "LIVERPOOL:",
    "SUBANG:",
    "MILAN:",
    "London:",
    "SERDANG:",
    "MONTREAL:",
    "KLUANG:",
    "BELGRADE:",
    "JAKARTA:",
    "MOSCOW:",
    "Naypyitaw:",
    "MANCHESTER:",
    "STOCKHOLM:",
    "KLANG:",
    "SEPANG:",
    "BRUSSELS:",
    "WENTWORTH:",
    "LANGKAWI:",
    "REMBAU:",
    "SINGAPORE:",
    "ODENSE:",
    "209:",
    "PALEMBANG:",
    "TAIPING:",
    "PARIS:",
    "SEREMBAN:",
    "WASHINGTON:",
    "KABUL:",
    "LAUSANNE:",
    "KAJANG:",
    "KUANTAN:",
    "LYON:",
    "SIBU:",
    "GUANGZHOU:",
    "JOHANNESBURG:",
    "MONACO:",
    "PEKAN:",
    "VERSAILLES:",
    "KARACHI:",
    "PHUKET:",
    "MADRID:",
    "KINGSTON:",
    "NUERBURGRING:",
    "IPOH:",
    "KANGAR:",
    "SHANGHAI:",
    "GLASGOW:",
    "ROME:",
    "MEMPHIS:",
    "YANGON:",
    "ZURICH:",
    "MIAMI:",
    "PENANG:",
    "MUNICH:",
    "SILVERSTONE:",
    "PHOENIX:",
    "PONTIAN:",
    "NYAPYITAW:",
    "SELAYANG:",
    "TORONTO:",
    "MARANG:",
    "ZAGREB:",
    "MELBOURNE:",
    "NAYPYITAW:",
    "LONDON:",
    "NICOSIA:",
    "EXCLUSIVE:",
    "SALVADOR:",
    "BANGKOK:",
    "MANILA:",
    "PERAI:",
    "DUBAI:",
    "INDIANAPOLIS:",
    "Name:",
    "KUALALUMPUR:",
    "BUDAPEST:",
    "BARCELONA:",
    "BEIJING:",
    "COPENHAGEN:",
    "KANAGR:",
    "ISTANBUL:",
    "BRASILIA:",
    "MALACCA:",
    "VIENNA:",
    "USEFUL:",
    "PENANG:",
    "BERLIN:",
    "PAYERNE:",
    "DALLAS:",
    "Update:",
    "SYDNEY:",
    "GAZA:",
    "YANGON:",
    "KAWASAKI:",
    "CHICAGO:",
    "BOGOTA:",
    "REVAMP:",
    "ANKARA:",
    "MINISCULE:",
    "DAKAR:",
    "WARSAW:",
    "SOFIA:",
    "KUNSHAN:",
    "MILAN:",
    "LAGOS:",
    "Hagerman:",
    "HOMESTEAD:",
    "MONTREAL:",
    "Warren:",
    "TASHKENT:",
    "Howard:",
    "agic:",
    "JAKARTA:",
    "MOSCOW:",
    "WARWICK:",
    "HANOI:",
    "DETROIT:",
    "Omidyar:",
    "STOCKHOLM:",
    "AMMAN:",
    "BRUSSELS:",
    "BUDAPEST:",
    "SINGAPORE:",
    "DENVER:",
    "CINZANA:",
    "AMSTERDAM:",
    "GENEVA:",
    "PARIS:",
    "WASHINGTON:",
    "UPDATE:",
    "KABUL:",
    "YOKOHAMA:",
    "BUCHAREST:",
    "JERUSALEM:",
    "OSAKA:",
    "OTTAWA:",
    "JOHANNESBURG:",
    "inFamous:",
    "MONACO:",
    "TAIPEI:",
    "DHAKA:",
    "SANTIAGO:",
    "MADRID:",
    "UNCONSTITUTIONAL:",
    "PHILADELPHIA:",
    "TEHRAN:",
    "EDINBURGH:",
    "Updated:",
    "SEOUL:",
    "SHANGHAI:",
    "MINNEAPOLIS:",
    "BENDY:",
    "ROME:",
    "ATLANTA:",
    "SEATTLE:",
    "Infamous:",
    "CARACAS:",
    "ZURICH:",
    "MIAMI:",
    "ISLAMABAD:",
    "SOCHI:",
    "MUNICH:",
    "HAVANA:",
    "FRANKFURT:",
    "BEIJING:",
    "RIGA:",
    "WELLINGTON:",
    "OSLO:",
    "TALLINN:",
    "TORONTO:",
    "KAMPALA:",
    "PITTSBURGH:",
    "BOSTON:",
    "MELBOURNE:",
    "REUTERS:",
    "LONDON:",
    "NAIROBI:",
    "HELSINKI:",
    "TOKYO:",
    "LUXEMBOURG:",
    "BANGKOK:",
    "SARAJEVO:",
    "MANILA:",
    "HAMBURG:",
    "Desouza:",
    "SHENZHEN:",
    "DUBAI:",
    "BARCELONA:",
    "Wong:",
    "COPENHAGEN:",
    "KINSHASA:",
    "FRANCISCO:",
    "ISTANBUL:",
    "BRASILIA:",
    "MUMBAI:",
    "BAGHDAD:",
    "BANGALORE:",
    "VIENNA:",
    "NEW YORK:",
    "Haiti ",
    "PARIS ",
    "Malaysia ",
    "MINNEAPOLIS ",
    "Hawaii ",
    "MIAMI ",
    "JANEIRO ",
    "Indonesia ",
    "Bank ",
    "NASHVILLE ",
    "Denmark ",
    "PHILADELPHIA ",
    "Jamaica ",
    "AMMAN ",
    "LONDON ",
    "CHARLOTTE ",
    "FRANCISCO ",
    "YORK ",
    "DUBLIN ",
    "Washington ",
    "BEIJING ",
    "BERLIN ",
    "SEATTLE ",
    "THAILAND ",
    "KENYA ",
    "Utah ",
    "of ",
    "Ireland ",
    "Bhutan ",
    "Morocco ",
    "ANGELES ",
    "BALTIMORE ",
    "Jordan ",
    "Syria ",
    "Brazil ",
    "ORONTO ",
    "Texas ",
    "BOSTON ",
    "Canada ",
    "ROME ",
    "England ",
    "LOUIS ",
    "Myanmar ",
    "TOWN ",
    "Iowa ",
    "ORLANDO ",
    "CITY ",
    "Israel ",
    "BANGKOK ",
    "CLEVELAND ",
    "Afghanistan ",
    "Kenya ",
    "Zambia ",
    "TUNISIA ",
    "Spain ",
    "Iraq ",
    "HOUSTON ",
    "YouTubeMOSCOW ",
    "India ",
    "INDIANAPOLIS ",
    "ATLANTA ",
    "BEIRUT ",
    "CAIRO ",
    "ISTANBUL ",
    "Italy ",
    "GENEVA ",
    "CHICAGO ",
    "France ",
    "ISLAMABAD ",
    "Service ",
    "DELHI ",
    "Pakistan ",
    "Tanzania ",
    "Mexico ",
    "Germany ",
    "Ohio ",
    "HONOLULU ",
    "Ukraine ",
    "WASHINGTON ",
    "Liberia ",
    "ORLEANS ",
    "JERUSALEM ",
    "TORONTO ",
]

ADDITIONAL_STOP_WORDS = [
    "also",
    "uk",
    "www",
    "html",
    "and",
    "reuters",
    "theguardian",
    "washington",
    "ap",
    "afp",
    "relaxnews",
    "nbcnews",
    "nbc",
    "bbc",
    "rtrfaithworld",
    "faithworld",
    "rns",

    "mr",
    "ms",
    "f",
    "v",
    "k",
    "n",
    "o",
    "ll",
    "walletpop",

    # people name
    "kj",
    "dellantonia",
    "bengaluru",
    "uk",
    "pct",

    # Nation
    "lewes",
    "england",
    "us",
    "u",
    "un",
    "ul",
    "lo",

    # Place in California
    "pasadena",
    "madison",
    "dallas",
    "monaco",

    # Unnecessary words
    "yesterday",
    "ys",
    "mg",
    "end",
    "nzwili",
    'amb',
    'salkin',
    'gibson'
]


##################################
# Extend stop words              #
#################################
STOP_WORDS = corpus.stopwords.words("english")
TOKENIZER = RegexpTokenizer(r'\w+')
STOP_WORDS.extend(ADDITIONAL_STOP_WORDS)

def strip_punctuation(str):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub('', str)
    # return str.translate(string.maketrans("", ""), string.punctuation)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = string.lower()
    for replace_word in REPLACEMENT_WORDS:
        string = string.replace(replace_word, " ")

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    string = strip_punctuation(string)

    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def normalize_to_unicode(text):
    if type(text) is not unicode:  # If the input is already unicode, no need to decode again
        try:
            text = text.decode('utf-8')  # Might throw exception on cp1252 encoding
        except UnicodeDecodeError:
            text = text.decode('cp1252')  # This should be the most plausibly correct encoding, since it's not UTF8
                                          # Note that cp1252 will not fail on any input

    # These cases happen only when we are given unicode as input
    # This is because \xNN can happen only with 'charmap' decoding, which we don't do above
    text = text.replace(u'\x93', '"')
    text = text.replace(u'\x94', '"')
    text = text.replace(u'\x96', '-')
    text = text.replace(u'\x97', '-')
    text = text.replace(u'\x91', "'")
    text = text.replace(u'\x92', "'")
    text = text.replace(u'\xa0', ' ')
    text = re.sub('[\x80-\xff]', '', text)

    # These cases are properly encoded unicode characters (either UTF-8 or cp1252)
    text = text.replace(u'\u201c', '"')
    text = text.replace(u'\u201d', '"')
    text = text.replace(u'\u2013', '-')
    text = text.replace(u'\u2014', '-')
    text = text.replace(u'\u2018', "'")
    text = text.replace(u'\u2019', "'")

    # Return the text as unicode
    return text


from nltk import corpus
from nltk.tokenize import RegexpTokenizer
from HTMLParser import HTMLParser
import re

ADDITIONAL_STOP_WORDS = [
    "also",
    "uk",
    "www",
    "html",
    "and",
    "reuters",
    "theguardian",
    "washington",
    "ap",
    "afp",
    "relaxnews",
    "nbcnews",
    "nbc",
    "bbc",
    "rtrfaithworld",
    "faithworld",
    "rns",

    "mr",
    "ms",
    "f",
    "v",
    "k",
    "n",
    "o",
    "ll",
    "walletpop",

    # people name
    "kj",
    "dellantonia",
    "bengaluru",
    "uk",
    "pct",

    # Nation
    "lewes",
    "england",
    "us",
    "u",
    "un",
    "ul",
    "lo",

    # Place in California
    "pasadena",
    "madison",
    "dallas",
    "monaco",

    # Unnecessary words
    "yesterday",
    "ys",
    "mg",
    "end",
    "nzwili",
    'amb',
    'salkin',
    'gibson'
]

##################################
# Extend stop words              #
#################################
STOP_WORDS = corpus.stopwords.words("english")
STOP_WORDS.extend(ADDITIONAL_STOP_WORDS)

def contain_digit(string):
    for e in string:
        if e.isdigit():
            return True
    return False

# List of contractions adapted from Robert MacIntyre's tokenizer.
CONTRACTIONS2 = [re.compile(r"(?i)\b(can)(not)\b"),
                 re.compile(r"(?i)\b(d)('ye)\b"),
                 re.compile(r"(?i)\b(gim)(me)\b"),
                 re.compile(r"(?i)\b(gon)(na)\b"),
                 re.compile(r"(?i)\b(got)(ta)\b"),
                 re.compile(r"(?i)\b(lem)(me)\b"),
                 re.compile(r"(?i)\b(mor)('n)\b"),
                 re.compile(r"(?i)\b(wan)(na) ")]
CONTRACTIONS3 = [re.compile(r"(?i) ('t)(is)\b"),
                 re.compile(r"(?i) ('t)(was)\b")]
CONTRACTIONS4 = [re.compile(r"(?i)\b(whad)(dd)(ya)\b"),
                 re.compile(r"(?i)\b(wha)(t)(cha)\b")]

from nltk.stem.porter import *

def tokenize(text):
    """Regex-based tokenization. Modified from nltk TreebankWordTokenizer to handle left single quote.
    """
    # starting quotes
    text = re.sub(r'^(\"|\'\')', r'``', text)
    text = re.sub(r'(``)', r' \1 ', text)
    text = re.sub(r'([ (\[{<])("|\'\')', r'\1 `` ', text)
    text = re.sub(r"^'", r'` ', text)
    text = re.sub(r"([ (\[{<])'", r'\1 ` ', text)

    # punctuation
    text = re.sub(r'([:,])([^\d])', r' \1 \2', text)
    text = re.sub(r'\.\.\.', r' ... ', text)
    text = re.sub(r'[;@#$%&]', r' \g<0> ', text)
    text = re.sub(r'([^\.])(\.)([\]\)}>"\']*)\s*$', r'\1 \2\3 ', text)
    text = re.sub(r'[?!]', r' \g<0> ', text)

    # Tokenize left single quote
    text = re.sub(r"([^' ])('[sS]|'[mM]|'[dD]|') ", r"\1 \2 ", text)
    text = re.sub(r"([^' ])('ll|'re|'ve|n't|) ", r"\1 \2 ", text)
    text = re.sub(r"([^' ])('LL|'RE|'VE|N'T|) ", r"\1 \2 ", text)
    text = re.sub(r" '((?!['tTnNsSmMdD] |\s|[2-9]0s |em |till |cause |ll |LL |ve |VE |re |RE )\S+)", r" ` \1", text)

    # parens, brackets, etc.
    text = re.sub(r'[\]\[\(\)\{\}\<\>]', r' \g<0> ', text)
    text = re.sub(r'--', r' -- ', text)

    # add extra space to make things easier
    text = " " + text + " "

    # ending quotes
    text = re.sub(r'"', " '' ", text)
    text = re.sub(r"''", " '' ", text)
    text = re.sub(r'(\S)(\'\')', r'\1 \2 ', text)

    for regexp in CONTRACTIONS2:
        text = regexp.sub(r' \1 \2 ', text)
    for regexp in CONTRACTIONS3:
        text = regexp.sub(r' \1 \2 ', text)

    # We are not using CONTRACTIONS4 since
    # they are also commented out in the SED scripts
    # for regexp in self.CONTRACTIONS4:
    #     text = regexp.sub(r' \1 \2 \3 ', text)

    text = text.strip()

    res = text.split()
    stemmer = PorterStemmer()
    res = [stemmer.stem(e) for e in res if e not in STOP_WORDS and not contain_digit(e) and len(e) < 15]
    return res
