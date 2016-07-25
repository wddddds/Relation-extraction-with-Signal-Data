import json
import pandas as pd
import utils
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import pipeline, grid_search
from utils import cust_regression_vals, cust_txt_col, fmean_squared_error
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import make_scorer


with open('/Users/kaka/Desktop/ML/msc_project/data/labeled_data607.json') as gdata:
    data = gdata.readlines()


num_train = 20000
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
for article in data:
    current_article = json.loads(article)
    stanford_entities = current_article['stanford-entities']
    person, organisation = utils.group_entities(stanford_entities)
    for p in person:
        for o in organisation:
            data_pairs.append([p, o])
            data_content.append(current_article['content'].encode('utf-8'))
            article_index.append(current_article['article_index'])
            person_offset.append((p['start-offset'], p['end-offset']))
            organisation_offset.append((o['start-offset'], o['end-offset']))
            try:
                if p == current_article['new_ceo'] and o == current_article['org_appointed_new_ceo']:
                    data_label.append(1)

                else:
                    data_label.append(0)
            except KeyError:
                data_label.append(0)

print 'number of data pairs: ', len(data_pairs)

df = pd.DataFrame(data_pairs, columns=['PER', 'ORG'])
# print df.head(20)

label = pd.DataFrame(data_label, columns=['label'])
article_id = pd.DataFrame(article_index, columns=['article_id'])
content = pd.DataFrame(data_content, columns=['content'])
person_offset = pd.DataFrame(person_offset, columns=['person_start_offset', 'person_end_offset'])
organisation_offset = pd.DataFrame(organisation_offset,
                                   columns=['organisation_start_offset', 'organisation_end_offset'])
df = pd.concat([df, label, content, person_offset, organisation_offset, article_id], axis=1)
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
df['inside_content_words'] = df.apply(lambda x: utils.tokenize(x['inside_content']), axis=1)
df['entities_words_distance'] = df.apply(lambda x: len(x['inside_content_words']), axis=1).astype(np.int64)

print df.head(200)


# Train the model
df_train = df.iloc[:num_train]
df_test = df.iloc[num_train:]
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
                        ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='content')),
                        ('word_vector', vectorizer), ('tsvd1', tsvd)])),
                        ],
                    transformer_weights={'cst': 1.0, 'txt1': 0.5}
        )), ('rfr', rfc)])
param_grid = {'rfr__max_features': [10], 'rfr__max_depth': [20]}
model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=1, cv=5, verbose=20, scoring=RMSE)
model.fit(X_train, y_train)

prediction = model.predict(X_test)
pd.DataFrame({"id": id_test, "class": prediction}).to_csv('submission.csv', index=False)


result = pd.read_csv('submission.csv')
truth = df_test['label'].values

w = 0
extracted = 0
for i in xrange(7117):
    if result['class'].iloc[i] != truth[i]:
        w += 1
    if result['class'].iloc[i] == 1:
        extracted += 1

print 'how many wrong?: ', w
print 'how many extracted? :', extracted
