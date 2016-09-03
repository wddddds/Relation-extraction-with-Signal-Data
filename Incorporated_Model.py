import json
import pandas as pd
import utils
import numpy as np
import nltk
import random
import re
import textwrap
# import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import pipeline, grid_search
from utils import cust_regression_vals, cust_txt_col, fmean_squared_error
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import make_scorer
from collections import defaultdict

'''
===================================================================
                       rule based method
===================================================================
'''
with open('/Users/kaka/Desktop/ML/msc_project/data/labeled_data500_corrected.json') as gdata:
    data = gdata.readlines()
    data = data[0:2000]

knowledge_base = defaultdict(lambda: 0)

pattern_1 = re.compile("( ceo.*)")
pattern_2 = re.compile("(.* ceo of .*)")

newceo_org_per_pattern = re.compile("(.*)(has|named|appoint|names|tapped|announced|announces|appoints|"
                                    "confirmed the appointment|the appointment of a new ceo)(.*)")
newceo_org_per_end_pattern = re.compile("(.*)(CEO|chief executive officer|ceo|chief executive)(.*)")
newceo_org_per_pattern_no_end = re.compile("(.*)(announced|announces)(.*)(appointment)(.*)(new CEO)(.*)")
newceo_per_org_pattern = re.compile("(.*)(appointed|named|tapped|is|will step in as|takes over as)"
                                    "(.*)(ceo|chief executive officer|CEO)(.*)")
newceo_org_per_claim_pattern = re.compile("(.*)(has|appointed|named|tapped|announced)(.*)(ceo|"
                                          "CEO|chief executive officer)(.*)")
pattern1 = re.compile("(.&)(:)( )")
pattern1_cont = re.compile("(.*)(becomes new ceo)(.*)")
pattern2 = re.compile("(.&)(\*)( )")
pattern2_cont = re.compile("(.*)(replaces)(.*)(as ceo)")
pattern3 = re.compile("(.*)(has become the new ceo)(.*)")
pattern4 = re.compile("(.*the ceo of the.*)")
pattern4_before = re.compile("(.* named )")
pattern5 = re.compile("(.*)(is the new ceo of)(.*)")
pattern6 = re.compile("(.*)(confirmed|claimed|declared)(.*)")
pattern6_end = re.compile("(.*)(will become|is the|)(.*)(ceo|chief executive officer)(.*)")
pattern7 = re.compile("(.*)(says|claimed|declared)(.*)")
pattern7_end = re.compile("(.*)(has|was|been)(named|appointed|)(.*)(ceo|chief executive officer)(.*)")
pattern8 = re.compile("(.*)(acknowledged)(.*)(new ceo will be)(.*)")


per_org_pattern_ext = 0
org_per_pattern_ext = 0

correct = 0
wrong = 0

data_ind = 0
for article in data:
    org = 'None'
    new = 'None'
    found = 0
    current_article = json.loads(article)
    content = utils.pre_process_content(current_article['content'])
    stanford_entity = current_article['stanford-entities']
    signal_entity = current_article['signal-entities']
    stanford_person, stanford_organization = utils.group_entities(stanford_entity)
    signal_person, signal_organization = utils.group_signal_entities(signal_entity)
    for p in stanford_person:
        # print p
        # print p['start-offset']
        start_p = int(p['start-offset'])
        end_p = int(p['end-offset'])
        for o in stanford_organization:
            start_o = int(o['start-offset'])
            end_o = int(o['end-offset'])

            if o['surface-form'] not in ['Newstex', 'Bloomberg', 'ABC']:
                not_news_agent = True
            else:
                not_news_agent = False

            try:
                if content[start_o-1] != '(' and content[end_o+1] != ')':
                    no_bracket = True
                else:
                    no_bracket = False
            except IndexError:
                no_bracket = True

            if start_o - end_p < 50 and found == 0 and no_bracket and not_news_agent:
                if pattern4.match(content[end_p:start_o]) and pattern4_before.match(content[start_p-20:start_p]):
                    knowledge_base[(o['surface-form'], p['surface-form'])] += 1
                    found = 1
                    per_org_pattern_ext += 1
                    org = o['surface-form']
                    new = p['surface-form']

            if start_p - end_o < 15 and found == 0 and no_bracket and not_news_agent:
                if pattern3.match(content[end_p:end_p+40]):
                    knowledge_base[(o['surface-form'], p['surface-form'])] += 1
                    found = 1
                    org_per_pattern_ext += 1
                    org = o['surface-form']
                    new = p['surface-form']

            if start_p - end_o < 95 and found == 0 and no_bracket and not_news_agent and not\
                    pattern4.match(content[end_p:end_p+30]):
                # if data['_source']['content'][end_o:start_p] == ' CEO ':
                if newceo_org_per_pattern_no_end.match(content[end_o:start_p])\
                    or newceo_org_per_pattern.match(content[end_o:start_p])\
                        and newceo_org_per_end_pattern.match(content[start_p:start_p+50])\
                    or newceo_org_per_claim_pattern.match(content[end_o:start_p])\
                    or pattern1.match(content[end_o:start_p])\
                        and pattern1_cont.match(content[start_p:start_p+50])\
                    or start_p - end_o < 10\
                        and pattern2_cont.match(content[start_p:start_p+50]):
                    knowledge_base[(o['surface-form'], p['surface-form'])] += 1
                    found = 1
                    org_per_pattern_ext += 1
                    org = o['surface-form']
                    new = p['surface-form']

            if start_o - end_p < 60 and found == 0 and no_bracket and not_news_agent:
                # if data['_source']['content'][end_o:start_p] == ' CEO ':
                if newceo_per_org_pattern.match(content[end_p:start_o]):
                    knowledge_base[(o['surface-form'], p['surface-form'])] += 1
                    found = 1
                    per_org_pattern_ext += 1
                    org = o['surface-form']
                    new = p['surface-form']

    if found == 1:
        try:
            if org.lower() in current_article['org_appointed_new_ceo']['surface-form'].lower() and new.lower() in \
                    current_article['new_ceo']['surface-form'].lower()\
                    or current_article['org_appointed_new_ceo']['surface-form'].lower() in org.lower() and \
                    current_article['new_ceo']['surface-form'].lower() in new.lower():
                correct += 1
                # print '='*100
                # for s in textwrap.wrap(current_article['content'], width=120):
                #     print s
                # print '*********CORRECT*********'
            else:
                wrong += 1
                print '='*100
                print current_article['article_index']
                for s in textwrap.wrap(current_article['content'], width=120):
                    print s
                print '=====', org, '====',  new
                print 'should be: ', 'ORG: ', current_article['org_appointed_new_ceo']['surface-form'], \
                    'PER', current_article['new_ceo']['surface-form']
                print '*********WRONG*********'
                for entity in current_article['stanford-entities']:
                    if entity['position'] == 'content':
                        print entity['type'], entity['surface-form']
                if current_article['org_appointed_new_ceo']['end-offset'] < current_article['new_ceo']['start-offset']:
                    print content[current_article['org_appointed_new_ceo']
                    ['end-offset']:current_article['new_ceo']['start-offset']]
                else:
                    print content[current_article['new_ceo']['end-offset']
                    :current_article['org_appointed_new_ceo']['start-offset']]
        except KeyError:
            wrong += 1
    #
    # if found == 0:
    #     if current_article['exist_relation'] == 'y':
    #         print '='*100
    #         for s in textwrap.wrap(current_article['content'], width=120):
    #             print s
    #         print '*********NOT-FOUND*********'
    #         for entity in current_article['stanford-entities']:
    #             if entity['position'] == 'content':
    #                 print entity['type'], entity['surface-form']

    data_ind += 1

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


'''
===================================================================
                       ml based method
===================================================================
'''

with open('/Users/kaka/Desktop/ML/msc_project/data/labeled_data500_corrected.json') as gdata:
    data = gdata.readlines()

num_train = 12000
negative_limit_prob = 0.05
number_of_words_nearby = 6
number_of_tokens_nearby = 4
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
                        org.append(current_article['org_appointed_new_ceo']['surface-form'].encode('utf-8'))
                        per.append(current_article['new_ceo']['surface-form'].encode('utf-8'))
                        amount_of_training_data += 1

                    else:
                        if random.random() < negative_limit_prob:
                            data_label.append(0)
                            data_pairs.append([p, o])
                            data_content.append(current_article['content'])
                            article_index.append(current_article['article_index'])
                            person_offset.append((p['start-offset'], p['end-offset']))
                            organisation_offset.append((o['start-offset'], o['end-offset']))
                            org.append(current_article['org_appointed_new_ceo']['surface-form'].encode('utf-8'))
                            per.append(current_article['new_ceo']['surface-form'].encode('utf-8'))
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
                        org.append('null'.encode('utf-8'))
                        per.append('null'.encode('utf-8'))
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
                        org.append(current_article['org_appointed_new_ceo']['surface-form'].encode('utf-8'))
                        per.append(current_article['new_ceo']['surface-form'].encode('utf-8'))

                    else:
                        data_label.append(0)
                        data_pairs.append([p, o])
                        data_content.append(current_article['content'])
                        article_index.append(current_article['article_index'])
                        person_offset.append((p['start-offset'], p['end-offset']))
                        organisation_offset.append((o['start-offset'], o['end-offset']))
                        org.append(current_article['org_appointed_new_ceo']['surface-form'].encode('utf-8'))
                        per.append(current_article['new_ceo']['surface-form'].encode('utf-8'))

                except KeyError:
                    data_label.append(0)
                    data_pairs.append([p, o])
                    data_content.append(current_article['content'])
                    article_index.append(current_article['article_index'])
                    person_offset.append((p['start-offset'], p['end-offset']))
                    organisation_offset.append((o['start-offset'], o['end-offset']))
                    org.append('null'.encode('utf-8'))
                    per.append('null'.encode('utf-8'))

print 'number of total amount entity pairs', amount_of_data
print 'number of training data pairs', amount_of_training_data
print 'number of data pairs: ', len(data_pairs)


'''
===================================================================
                      feature creation
===================================================================
'''

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
print '===creating distance features==='
df['entities_char_distance'] = df.apply(lambda x: min(abs(x['person_end_offset'] - x['organisation_start_offset']),
                                                      abs(x['organisation_end_offset'] - x['person_start_offset'])),
                                        axis=1).astype(np.int64)
print '===creating bag of words features==='
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
print '===creating POS features 1/6==='
df['content_pos'] = df.apply(lambda x: [s[1].encode('utf-8') for s in nltk.pos_tag(x['content_words'])], axis=1)
df.to_csv('POS_data_frame.csv')
print '===creating POS features 2/6==='
df['content_pos_str'] = df.apply(lambda x: ' '.join(x['content_pos']), axis=1)
print '===creating POS features 3/6==='
df['per_before_words_pos'] = df.apply(lambda x: 'null'.encode('utf-8') if x['per_entity_index'] < number_of_tokens_nearby
                                 else ' '.join(x['content_pos'][x['per_entity_index']-number_of_tokens_nearby:
                                 x['per_entity_index']]).encode('utf-8'), axis=1)
print '===creating POS features 4/6==='
df['per_after_words_pos'] = df.apply(lambda x: 'null'.encode('utf-8')
                                 if len(x['content_words'])-x['per_entity_index'] < number_of_tokens_nearby
                                 else ' '.join(x['content_pos'][x['per_entity_index']:
                                 x['per_entity_index']+number_of_tokens_nearby]).encode('utf-8'), axis=1)
print '===creating POS features 5/6==='
df['org_before_words_pos'] = df.apply(lambda x: 'null'.encode('utf-8') if x['org_entity_index'] < number_of_tokens_nearby
                                 else ' '.join(x['content_pos'][x['org_entity_index']-number_of_tokens_nearby:
                                 x['org_entity_index']]).encode('utf-8'), axis=1)
print '===creating POS features 6/6==='
df['org_after_words_pos'] = df.apply(lambda x: 'null'.encode('utf-8')
                                 if len(x['content_words'])-x['org_entity_index'] < number_of_tokens_nearby
                                 else ' '.join(x['content_pos'][x['org_entity_index']:
                                 x['org_entity_index']+number_of_tokens_nearby]).encode('utf-8'), axis=1)


df = pd.read_csv('data/ML_data_frame.csv')
# Train the model

df_train = df.iloc[:amount_of_training_data]
df_test = df.iloc[amount_of_training_data:]
id_test = df_test['article_id']
y_train = df_train['label'].values
X_train = df_train[:]
X_test = df_test[:]

vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1))

print '===creating vectorized bag of words features==='
# use random forest to train our model
rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42, verbose=1)
# add word_vector term to our feature(by sk-learn build in function)
tsvd = TruncatedSVD(n_components=5, random_state=42)
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
                        ('txt7', pipeline.Pipeline([('s7', cust_txt_col(key='content_pos_str')),
                        ('word_vector', vectorizer), ('tsvd7', tsvd)])),
                        ('txt8', pipeline.Pipeline([('s8', cust_txt_col(key='per_before_words_pos')),
                        ('word_vector', vectorizer), ('tsvd8', tsvd)])),
                        ('txt9', pipeline.Pipeline([('s9', cust_txt_col(key='org_before_words_pos')),
                        ('word_vector', vectorizer), ('tsvd9', tsvd)])),
                        ('txt10', pipeline.Pipeline([('s10', cust_txt_col(key='per_after_words_pos')),
                        ('word_vector', vectorizer), ('tsvd10', tsvd)])),
                        ('txt11', pipeline.Pipeline([('s11', cust_txt_col(key='org_after_words_pos')),
                        ('word_vector', vectorizer), ('tsvd11', tsvd)])),
                        ],
                    transformer_weights={'cst': 1.0, 'txt1': 0.5, 'txt2': 0.5, 'txt3': 0.5,
                                         'txt4': 0.5, 'txt5': 0.5, 'txt6': 0.5, 'txt7': 0.5, 'txt8': 0.5,
                                         'txt9': 0.5, 'txt10': 0.5, 'txt11': 0.5}
        )), ('rfr', rfc)])

'''
===================================================================
                      training ML model
===================================================================
'''

param_grid = {'rfr__max_features': [30], 'rfr__max_depth': [20]}
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
        if df_test['org'] in knowledge_base.keys():
            if knowledge_base[df_test['org']] == df_test['per']:
                corr += 1
        else:
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

print corr
print extracted
print 'final precision', doc_precision
print 'final recall', recall
print 'final F score', doc_f_socre
