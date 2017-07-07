from nltk import corpus
from nltk.tokenize import RegexpTokenizer
from HTMLParser import HTMLParser
import re
import string
from nltk.stem.porter import *

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


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []

    def handle_data(self, data):
        self.fed.append(data)

    def get_data(self):
        return "".join(self.fed)


class NormalizationText():
    @staticmethod
    def remove_html_tag(text):
        """
        Remove html tag in text. For example, <html>, <a>, <div>
        :param text:
        :return:
        """
        if text:
            s = MLStripper()
            s.feed(text)
            return s.get_data()
        else:
            return ""

    @staticmethod
    def normalize_to_unicode(text):
        if type(text) is not unicode:  # If the input is already unicode, no need to decode again
            try:
                text = text.decode('utf-8')  # Might throw exception on cp1252 encoding
            except UnicodeDecodeError:
                text = text.decode('cp1252')  # This should be the most plausibly correct encoding, since it's not UTF8
                                              # Note that cp1252 will not fail on any input

        # These cases happen only when we are given unicode as input
        # This is because \xNN can happen only with 'charmap' decoding, which we don't do above
        text = text.replace(u'\x19',"'")
        text = text.replace(u'\x14',"")

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

    @staticmethod
    def preprocess(text):
        def strip_punctuation(str):
            # deleted_not_dot = string.punctuation.replace('.', '')
            regex = re.compile('[%s]' % re.escape(string.punctuation))
            return regex.sub('', str)
        text = strip_punctuation(text)

        for replace_word in REPLACEMENT_WORDS:
            text = text.replace(replace_word, " ")

        text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
        text = re.sub(r"\'s", " \'s", text)
        text = re.sub(r"\'ve", " \'ve", text)
        text = re.sub(r"n\'t", " n\'t", text)
        text = re.sub(r"\'re", " \'re", text)
        text = re.sub(r"\'d", " \'d", text)
        text = re.sub(r"\'ll", " \'ll", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\(", " \( ", text)
        text = re.sub(r"\)", " \) ", text)
        text = re.sub(r"\?", " \? ", text)
        text = re.sub(r"\s{2,}", " ", text)

        text = NormalizationText.normalize_to_unicode(text).strip()
        stemmer = PorterStemmer()
        content = " ".join([stemmer.stem(w) for w in TOKENIZER.tokenize(text) if w.lower() not in STOP_WORDS and len(w) < 15])
        return content.lower()
