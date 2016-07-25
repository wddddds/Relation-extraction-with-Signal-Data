import json
import utils
import re
from collections import defaultdict

# fin = open('data/signalmedia-1M.jsonl')
mdata = open('/Users/kaka/Desktop/ML/msc_project/data/signalmedia-1m.jsonl')
jdata = open('/Users/kaka/Desktop/ML/msc_project/data/wetransfer-79f35e/articles-01092015.json')


with open('/Users/kaka/Desktop/ML/msc_project/data/articles-ceo.json') as ceo:
    ceodata = ceo.readlines()
    ceodata = ceodata[0:20000]

# cdata = json.loads(ceodata[13])
# print cdata['_source']['content'][144:146]
# print cdata['_source']['content']
# print cdata['_source']['stanford-entities']
# print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
# sample = json.dumps(cdata, indent=4, separators=(',', ':'))
# print sample

knowledge_base = defaultdict(lambda: 0)

pattern = re.compile("(( CEO)(.*))")
pattern2 = re.compile("((.*)( CEO of )(.*))")


for article in ceodata:
    data = json.loads(article)
    nedata = data['_source']['stanford-entities']
    person, organization = utils.group_entities(nedata)
    # print person
    # print organization
    for p in person:
        # print p
        # print p['start-offset']
        start_p = int(p['start-offset'])
        end_p = int(p['end-offset'])
        for o in organization:
            start_o = int(o['start-offset'])
            end_o = int(o['end-offset'])
            if start_p - end_o < 6:
                # if data['_source']['content'][end_o:start_p] == ' CEO ':
                if pattern.match(data['_source']['content'][end_o:start_p]):
                    knowledge_base[(o['text'], p['text'])] += 1
            if start_o - end_p < 30:
                # if data['_source']['content'][end_o:start_p] == ' CEO ':
                if pattern2.match(data['_source']['content'][end_p:start_o]):
                    knowledge_base[(o['text'], p['text'])] += 1

    # for oo in organization:
    #     # if oo['text'] == 'Google':
    #     #     print data['_source']['tracking-url']
    #     # print p
    #     # print p['start-offset']
    #     start_oo = int(oo['start-offset'])
    #     end_oo = int(oo['end-offset'])
    #     for pp in person:
    #         start_pp = pp['start-offset']
    #         end_pp = int(pp['end-offset'])
    #         if start_oo - end_pp < 30:
    #             # if data['_source']['content'][end_o:start_p] == ' CEO ':
    #             if pattern2.match(data['_source']['content'][end_pp:start_oo]):
    #                 knowledge_base[(oo['text'], pp['text'])] += 1

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

s = 0
for a, b in knowledge_base.keys():
    s += knowledge_base[a, b]
print 'total relation extracted :', s


print knowledge_base
print len(knowledge_base)
print len(Knowledge_base)

print Knowledge_base['Microsoft']
print Knowledge_base['Yahoo']
print Knowledge_base['Facebook']
print Knowledge_base['Google']
print Knowledge_base['Cisco Systems']
print Knowledge_base['Apple']
print Knowledge_base['Huawei']
print Knowledge_base['Alibaba']
