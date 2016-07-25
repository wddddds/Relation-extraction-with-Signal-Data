import re
import nltk

from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error


def group_entities(data):
    person = []
    organization = []
    for e in data:
        if e['position'] == 'content':
            if e['type'] == 'person':
                person.append(e)
            elif e['type'] == 'organization':
                organization.append(e)

    return person, organization


def group_signal_entities(data):
    person = []
    organization = []
    for e in data:
        if e['position'] == 'content':
            if e['signal-type'] == 'person':
                person.append(e)
            elif e['signal-type'] == 'organisation':
                organization.append(e)

    return person, organization


def tokenize(s):
    # s = s.lower()
    # s = re.sub(r'\n', r' ', s)
    # s = re.sub('[^a-zA-Z?!\'.,]', ' ', s)
    # s = s.replace("..", ".")
    # s = s.replace(r'.', r' . ')
    # s = s.replace(r',', r' , ')
    # s = s.replace('\u2019', '')
    # s = s.replace('  ', ' ')
    # s = s.split(' ')
    # stop = set(stopwords.words('english'))
    # s = [w for w in s if w not in stop and w != '']

    s = nltk.word_tokenize(s)

    return s


def pre_process_content(s):
    s = s.lower()

    return s


class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        d_col_drops = ['PER', 'label', 'ORG', 'content', 'content_words', 'org', 'per',
                       'inside_content', 'article_id', 'inside_content_words',
                       'per_before_words', 'org_before_words', 'per_after_words', 'org_after_words',
                       # 'content_pos', 'content_pos_str', 'per_before_words_pos', 'per_after_words_pos',
                       # 'org_before_words_pos', 'org_after_words_pos']
                       ]
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        return hd_searches


class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key].apply(str)


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_
