import json
import nltk
import textwrap

with open('/Users/kaka/Desktop/ML/msc_project/data/articles-ceo.json') as ceo:
    ceodata = ceo.readlines()
    ceodata = ceodata[0:1000]

# with open('/Users/kaka/Desktop/ML/msc_project/data/wetransfer-79f35e/articles-01092015.json') as gdata:
with open('/Users/kaka/Desktop/ML/msc_project/data/new_ceo/ceo_apr16.json') as gdata:
    data = gdata.readlines()

cdata = json.loads(data[9])
# print cdata['_source']['content'][144:146]
# print cdata['_source']['content']
# print cdata['_source']['stanford-entities']
print '%'*100
sample = json.dumps(cdata, indent=4, separators=(',', ':'))
print sample
content = cdata['content']
content_with_pos_tag = nltk.pos_tag(nltk.word_tokenize(content))
print '*'*100
print content
print content_with_pos_tag
print len(data)

for article in data[100:150]:
    c_article = json.loads(article)
    print '*'*100
    print '*'*100
    for s in textwrap.wrap(c_article['content'], width=80):
        print s
    print "********signal entities********"
    signal_en = (en for en in c_article['signal-entities'] if en['position'] == 'content')
    signal_en = list(signal_en)
    print json.dumps(signal_en, indent=4, separators=(',', ':'))
    print "********stanford entities********"
    stanford_en = (en for en in c_article['stanford-entities'] if en['position'] == 'content')
    stanford_en = list(stanford_en)
    print json.dumps(stanford_en, indent=4, separators=(',', ':'))
