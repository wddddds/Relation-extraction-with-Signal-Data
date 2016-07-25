import json
import pandas as pd
import utils
import numpy as np
import random
# import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import pipeline, grid_search
from utils import cust_regression_vals, cust_txt_col, fmean_squared_error
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import make_scorer
from collections import defaultdict


with open('/Users/kaka/Desktop/ML/msc_project/data/labeled_data500_corrected.json') as gdata:
    data = gdata.readlines()

num_train = 13000
negative_limit_prob = 0.33
number_of_words_nearby = 6
knowledge_base = defaultdict(lambda: 0)
RMSE = make_scorer(fmean_squared_error, greater_is_better=False)

'''
===================================================================
                       data manipulation
===================================================================
'''
data_pairs = []
data_label = []
data_content = []
article_index = []
person_offset = []
organisation_offset = []
org = []
per = []
amount_of_data = 0
amount_of_training_data = 0
training_article_amount = 0
to_be_extracted = 0
for article in data:
    current_article = json.loads(article)
    stanford_entities = current_article['stanford-entities']
    person, organisation = utils.group_entities(stanford_entities)
    for p in person:
        for o in organisation:
            try:
                if {
                     p['surface-form'].lower() in current_article['new_ceo']['surface-form'].lower() and
                     o['surface-form'].lower() in current_article['org_appointed_new_ceo']['surface-form'].lower()
                    or current_article['new_ceo']['surface-form'].lower() in p['surface-form'].lower() and
                     current_article['org_appointed_new_ceo']['surface-form'].lower() in o['surface-form'].lower()
                }:
                    to_be_extracted += 1
            except KeyError:
                pass

            amount_of_data += 1
            if amount_of_data < num_train:
                try:
                    if p == current_article['new_ceo'] and o == current_article['org_appointed_new_ceo']:
                        data_label.append(1)
                        data_pairs.append([p, o])
                        data_content.append(current_article['content'])
                        article_index.append(current_article['article_index'])
                        person_offset.append((p['start-offset'], p['end-offset']))
                        organisation_offset.append((o['start-offset'], o['end-offset']))
                        org.append(current_article['org_appointed_new_ceo']['surface-form'])
                        per.append(current_article['new_ceo']['surface-form'])
                        amount_of_training_data += 1

                    else:
                        if random.random() < negative_limit_prob:
                            data_label.append(0)
                            data_pairs.append([p, o])
                            data_content.append(current_article['content'])
                            article_index.append(current_article['article_index'])
                            person_offset.append((p['start-offset'], p['end-offset']))
                            organisation_offset.append((o['start-offset'], o['end-offset']))
                            org.append(current_article['org_appointed_new_ceo']['surface-form'])
                            per.append(current_article['new_ceo']['surface-form'])
                            amount_of_training_data += 1
                        else:
                            pass
                except KeyError:
                    if random.random() < negative_limit_prob:
                        data_label.append(0)
                        data_pairs.append([p, o])
                        data_content.append(current_article['content'])
                        article_index.append(current_article['article_index'])
                        person_offset.append((p['start-offset'], p['end-offset']))
                        organisation_offset.append((o['start-offset'], o['end-offset']))
                        org.append('null')
                        per.append('null')
                        amount_of_training_data += 1
            else:
                try:
                    if p == current_article['new_ceo'] and o == current_article['org_appointed_new_ceo']:
                        data_label.append(1)
                        data_pairs.append([p, o])
                        data_content.append(current_article['content'])
                        article_index.append(current_article['article_index'])
                        person_offset.append((p['start-offset'], p['end-offset']))
                        organisation_offset.append((o['start-offset'], o['end-offset']))
                        org.append(current_article['org_appointed_new_ceo']['surface-form'])
                        per.append(current_article['new_ceo']['surface-form'])

                    else:
                        data_label.append(0)
                        data_pairs.append([p, o])
                        data_content.append(current_article['content'])
                        article_index.append(current_article['article_index'])
                        person_offset.append((p['start-offset'], p['end-offset']))
                        organisation_offset.append((o['start-offset'], o['end-offset']))
                        org.append(current_article['org_appointed_new_ceo']['surface-form'])
                        per.append(current_article['new_ceo']['surface-form'])

                except KeyError:
                    data_label.append(0)
                    data_pairs.append([p, o])
                    data_content.append(current_article['content'])
                    article_index.append(current_article['article_index'])
                    person_offset.append((p['start-offset'], p['end-offset']))
                    organisation_offset.append((o['start-offset'], o['end-offset']))
                    org.append('null')
                    per.append('null')

print 'number of total amount entity pairs', amount_of_data
print 'number of training data pairs', amount_of_training_data
print 'number of data pairs: ', len(data_pairs)

df = pd.DataFrame(data_pairs, columns=['PER', 'ORG'])
# print df.head(20)

label = pd.DataFrame(data_label, columns=['label'])
Org = pd.DataFrame(org, columns=['org'])
Per = pd.DataFrame(per, columns=['per'])
article_id = pd.DataFrame(article_index, columns=['article_id'])
content = pd.DataFrame(data_content, columns=['content'])
person_offset = pd.DataFrame(person_offset, columns=['person_start_offset', 'person_end_offset'])
organisation_offset = pd.DataFrame(organisation_offset,
                                   columns=['organisation_start_offset', 'organisation_end_offset'])
df = pd.concat([df, label, content, person_offset, organisation_offset, article_id, Per, Org], axis=1)
df = pd.DataFrame(df)

df['content_words'] = df.apply(lambda x: utils.tokenize(x['content']), axis=1)
df['per_entity_index'] = df.apply(lambda x: len(utils.tokenize(x['content'][0:x['person_start_offset']])), axis=1)\
    .astype(np.int64)
df['org_entity_index'] = df.apply(lambda x: len(utils.tokenize(x['content']
                                                               [0:x['organisation_start_offset']])), axis=1)\
    .astype(np.int64)


# length feature
df['entities_char_distance'] = df.apply(lambda x: min(abs(x['person_end_offset'] - x['organisation_start_offset']),
                                                      abs(x['organisation_end_offset'] - x['person_start_offset'])),
                                        axis=1).astype(np.int64)
df['inside_content'] = df.apply(lambda x: x['content'][x['organisation_end_offset']:x['person_start_offset']]
                                if x['organisation_end_offset'] < x['person_start_offset']
                                else x['content'][x['person_end_offset']:x['organisation_start_offset']], axis=1)
df['inside_content'] = df.apply(lambda x: x['inside_content'].encode('utf-8'), axis=1)
df['inside_content_words'] = df.apply(lambda x: utils.tokenize(x['inside_content'].decode('utf-8')), axis=1)
df['entities_words_distance'] = df.apply(lambda x: len(x['inside_content_words']), axis=1).astype(np.int64)
df['per_before_words'] = df.apply(lambda x: 'null'.encode('utf-8') if x['per_entity_index'] < number_of_words_nearby
                                 else ' '.join(x['content_words'][x['per_entity_index']-number_of_words_nearby:
                                 x['per_entity_index']]).encode('utf-8'), axis=1)
df['per_after_words'] = df.apply(lambda x: 'null'.encode('utf-8')
                                 if len(x['content_words'])-x['per_entity_index'] < number_of_words_nearby
                                 else ' '.join(x['content_words'][x['per_entity_index']:
                                 x['per_entity_index']+number_of_words_nearby]).encode('utf-8'), axis=1)
df['org_before_words'] = df.apply(lambda x: 'null'.encode('utf-8') if x['org_entity_index'] < number_of_words_nearby
                                 else ' '.join(x['content_words'][x['org_entity_index']-number_of_words_nearby:
                                 x['org_entity_index']]).encode('utf-8'), axis=1)
df['org_after_words'] = df.apply(lambda x: 'null'.encode('utf-8')
                                 if len(x['content_words'])-x['org_entity_index'] < number_of_words_nearby
                                 else ' '.join(x['content_words'][x['org_entity_index']:
                                 x['org_entity_index']+number_of_words_nearby]).encode('utf-8'), axis=1)
df['content'] = df.apply(lambda x: x['content'].encode('utf-8'), axis=1)

# df = df.reindex(np.random.permutation(df.index))
# print df.head(200)


# Train the model
df_train = df.iloc[:amount_of_training_data]
df_test = df.iloc[amount_of_training_data:]
id_test = df_test['article_id']
y_train = df_train['label'].values
X_train = df_train[:]
X_test = df_test[:]

vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1))

# use random forest to train our model
rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42, verbose=1)
# add word_vector term to our feature(by sk-learn build in function)
tsvd = TruncatedSVD(n_components=10, random_state=42)
clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list=[
                        ('cst',  cust_regression_vals()),
                        ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='inside_content')),
                        ('word_vector', vectorizer), ('tsvd1', tsvd)])),
                        ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='content')),
                        ('word_vector', vectorizer), ('tsvd2', tsvd)])),
                        ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='per_before_words')),
                        ('word_vector', vectorizer), ('tsvd3', tsvd)])),
                        ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='org_before_words')),
                        ('word_vector', vectorizer), ('tsvd4', tsvd)])),
                        ('txt5', pipeline.Pipeline([('s5', cust_txt_col(key='per_after_words')),
                        ('word_vector', vectorizer), ('tsvd5', tsvd)])),
                        ('txt6', pipeline.Pipeline([('s6', cust_txt_col(key='org_after_words')),
                        ('word_vector', vectorizer), ('tsvd6', tsvd)])),
                        ],
                    transformer_weights={'cst': 1.0, 'txt1': 0.5, 'txt2': 0.5, 'txt3': 0.5,
                                         'txt4': 0.5, 'txt5': 0.5, 'txt6': 0.5}
        )), ('rfr', rfc)])
param_grid = {'rfr__max_features': [10], 'rfr__max_depth': [20]}
model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=1, cv=5, verbose=20, scoring=RMSE)
model.fit(X_train, y_train)

prediction = model.predict(X_test)
pd.DataFrame({"id": id_test, "class": prediction}).to_csv('submission.csv', index=False)


result = pd.read_csv('submission.csv')
truth = df_test['label'].values
# print org

'''
===================================================================
                  Knowledge base construction
===================================================================
'''

corr = 0
for i in xrange(len(X_test)):
    if result['class'].iloc[i] == 1:
        knowledge_base[(df_test['ORG'].iloc[i]['surface-form'], df_test['PER'].iloc[i]['surface-form'])] += 1
        try:
            print '=' * 100
            print df_test['PER'].iloc[i]['surface-form'].lower()
            print df_test['ORG'].iloc[i]['surface-form'].lower()
            print '=' * 30
            print df_test['per'].iloc[i].lower()
            print df_test['org'].iloc[i].lower()
            print '*' * 30
            if (
                 df_test['ORG'].iloc[i]['surface-form'].lower() in df_test['org'].iloc[i].lower() and
                 df_test['PER'].iloc[i]['surface-form'].lower() in df_test['per'].iloc[i].lower()
                        or df_test['org'].iloc[i].lower() in df_test['ORG'].iloc[i]
                 ['surface-form'].lower() and
                 df_test['per'].iloc[i].lower() in df_test['PER'].iloc[i]['surface-form'].lower()
            ):
                corr += 1
                print 'corr += 1'
        except (KeyError, TypeError):
            pass

Knowledge_base = {}
company_set_best = {}

for k, v in knowledge_base.iterkeys():
    if k not in company_set_best.keys():
        Knowledge_base[k] = v
        company_set_best[k] = knowledge_base[(k, v)]
    else:
        if knowledge_base[(k, v)] > company_set_best[k]:
            Knowledge_base[k] = v
            company_set_best[k] = knowledge_base[(k, v)]


w = 0
extracted = 0
w_e = 0
for i in xrange(len(X_test)):
    if result['class'].iloc[i] != truth[i]:
        w += 1
    if result['class'].iloc[i] == 1:
        extracted += 1
    # if result['']

for i in xrange(len(X_test)):
    if result['class'].iloc[i] != truth[i] and result['class'].iloc[i] == 1:
        w_e += 1

# print 'how many to be extract? :', (extracted-w_e)+(w-w_e)
# print 'how many wrong?: ', w
# print 'how many extracted? :', extracted
# print 'how many extracted and wrong? :', w_e
# print 'correctly extracted: ', extracted-w_e
# print 'doc_corr', corr

doc_precision = 1.0*corr/extracted
# doc_recall = 1.0*corr/to_be_extracted
precision = 1-(1.0*w_e/extracted)
recall = 1-(1.0*(w-w_e)/((extracted-w_e)+(w-w_e)))
F_score = 2*(precision * recall)/(precision + recall)
doc_f_socre = 2*(doc_precision * recall)/(doc_precision + recall)

# print 'precision: ', precision
# print 'recall: ', recall
# print 'F-score', F_score

# print corr
# print extracted
print 'final precision', doc_precision
print 'final recall', recall
print 'final F score', doc_f_socre
