import json
import utils
import re
import textwrap

from collections import defaultdict

with open('/Users/kaka/Desktop/ML/msc_project/data/labeled_data500.json') as gdata:
    data = gdata.readlines()

knowledge_base = defaultdict(lambda: 0)

pattern = re.compile("( CEO.*)")
pattern2 = re.compile("(.* CEO of .*)")

newceo_org_per_pattern = re.compile("(.*)(appoint|named|names|tapped|announced|"
                                    "announces|appoints|confirmed the appointment)(.*)")
new_pattern = re.compile("(.*)(has|named|appoint|names|tapped|announced|announces|appoints|"
                         "confirmed the appointment)(.*)")
newceo_org_per_end_pattern = re.compile("(.*)(CEO|chief executive officer|ceo|chief executive)(.*)")
newceo_org_per_pattern_no_end = re.compile("(.*)(announced|announces)(.*)(appointment)(.*)(new CEO)(.*)")
newceo_per_org_pattern = re.compile("(.*)(appointed|named|tapped|is)(.*)(ceo|chief executive officer|)(.*)")
newceo_org_per_claim_pattern = re.compile("(.*)(has|appointed|named|tapped|announced)(.*)(ceo|"
                                          "CEO|chief executive officer)(.*)")

per_org_pattern_ext = 0
org_per_pattern_ext = 0

data_ind = 0
for article in data:
    found = 0
    data = json.loads(article)
    stanford_entity = data['stanford-entities']
    signal_entity = data['signal-entities']
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
            if start_p - end_o < 50 and found == 0:
                # if data['_source']['content'][end_o:start_p] == ' CEO ':
                if newceo_org_per_pattern_no_end.match(data['content'][end_o:start_p])\
                    or new_pattern.match(data['content'][end_o:start_p])\
                        and newceo_org_per_end_pattern.match(data['content'][start_p:start_p+30])\
                        or newceo_org_per_claim_pattern.match(data['content'][end_o:start_p]):
                    knowledge_base[(o['surface-form'], p['surface-form'])] += 1
                    print '='*100
                    for s in textwrap.wrap(data['content'], width=120):
                        print s
                    print o['surface-form'], 'appoint', p['surface-form'], ' as new CEO'
                    found = 1
                    org_per_pattern_ext += 1
                    print 'data index: ', data_ind
                    try:
                        print data['content'][start_o-20:end_p+30]
                    except IndexError:
                        print data['content'][start_o:end_p]

            if start_o - end_p < 50 and found == 0:
                # if data['_source']['content'][end_o:start_p] == ' CEO ':
                if newceo_per_org_pattern.match(data['content'][end_p:start_o]):
                    knowledge_base[(o['surface-form'], p['surface-form'])] += 1
                    print '='*100
                    for s in textwrap.wrap(data['content'], width=120):
                        print s
                    print o['text'], 'appoint', p['surface-form'], ' as new CEO'
                    found = 1
                    per_org_pattern_ext += 1
                    print 'data index: ', data_ind
                    try:
                        print data['content'][start_p-20:end_o+30]
                    except IndexError:
                        print data['content'][start_p:end_o]
    #
    # if found == 0:
    #     person = stanford_person
    #     organization = signal_organization
    #     # print person
    #     # print organization
    #     for p in stanford_person:
    #         # print p
    #         # print p['start-offset']
    #         start_p = int(p['start-offset'])
    #         end_p = int(p['end-offset'])
    #         for o in signal_organization:
    #             start_o = int(o['offset'])
    #             end_o = start_o + len(o['surface-form'])
    #             if start_p - end_o < 50 and found == 0:
    #                 # if data['_source']['content'][end_o:start_p] == ' CEO ':
    #                 if newceo_org_per_pattern_no_end.match(data['content'][end_o:start_p])\
    #                     or new_pattern.match(data['content'][end_o:start_p])\
    #                         and newceo_org_per_end_pattern.match(data['content'][start_p:start_p+30])\
    #                         or newceo_org_per_claim_pattern.match(data['content'][end_o:start_p]):
    #                     knowledge_base[(o['surface-form'], p['surface-form'])] += 1
    #                     print '='*100
    #                     print 'content :', data['content']
    #                     print o['surface-form'], 'appoint', p['surface-form'], ' as new CEO'
    #                     found = 1
    #                     org_per_pattern_ext += 1
    #                     print 'data index: ', data_ind
    #                     try:
    #                         print data['content'][start_o-20:end_p+30]
    #                     except IndexError:
    #                         print data['content'][start_o:end_p]
    #
    #             if start_o - end_p < 50 and found == 0:
    #                 # if data['_source']['content'][end_o:start_p] == ' CEO ':
    #                 if newceo_per_org_pattern.match(data['content'][end_p:start_o]):
    #                     knowledge_base[(o['surface-form'], p['surface-form'])] += 1
    #                     print '='*100
    #                     print 'content :', data['content']
    #                     print o['surface-form'], 'appoint', p['surface-form'], ' as new CEO'
    #                     found = 1
    #                     per_org_pattern_ext += 1
    #                     print 'data index: ', data_ind
    #                     try:
    #                         print data['content'][start_p-20:end_o+30]
    #                     except IndexError:
    #                         print data['content'][start_p:end_o]
    #
    # if found == 0:
    #     person = signal_person
    #     organization = signal_organization
    #     # print person
    #     # print organization
    #     for p in signal_person:
    #         # print p
    #         # print p['start-offset']
    #         start_p = int(p['offset'])
    #         end_p = start_p + len(p['surface-form'])
    #         for o in signal_organization:
    #             start_o = int(o['offset'])
    #             end_o = start_o + len(o['surface-form'])
    #             if start_p - end_o < 50 and found == 0:
    #                 # if data['_source']['content'][end_o:start_p] == ' CEO ':
    #                 if newceo_org_per_pattern_no_end.match(data['content'][end_o:start_p])\
    #                     or new_pattern.match(data['content'][end_o:start_p])\
    #                         and newceo_org_per_end_pattern.match(data['content'][start_p:start_p+50])\
    #                         or newceo_org_per_claim_pattern.match(data['content'][end_o:start_p]):
    #                     knowledge_base[(o['surface-form'], p['surface-form'])] += 1
    #                     print '='*100
    #                     print 'content :', data['content']
    #                     print o['surface-form'], 'appoint', p['surface-form'], ' as new CEO'
    #                     found = 1
    #                     org_per_pattern_ext += 1
    #                     print 'data index: ', data_ind
    #                     try:
    #                         print data['content'][start_o-20:end_p+30]
    #                     except IndexError:
    #                         print data['content'][start_o:end_p]
    #
    #             if start_o - end_p < 50 and found == 0:
    #                 # if data['_source']['content'][end_o:start_p] == ' CEO ':
    #                 if newceo_per_org_pattern.match(data['content'][end_p:start_o]):
    #                     knowledge_base[(o['surface-form'], p['surface-form'])] += 1
    #                     print '='*100
    #                     print 'content :', data['content']
    #                     print o['surface-form'], 'appoint', p['surface-form'], ' as new CEO'
    #                     found = 1
    #                     per_org_pattern_ext += 1
    #                     print 'data index: ', data_ind
    #                     try:
    #                         print data['content'][start_p-20:end_o+30]
    #                     except IndexError:
    #                         print data['content'][start_p:end_o]

    if found == 0:
        if data['exist_relation'] == 'y':
            print '='*100
            for s in textwrap.wrap(data['content'], width=120):
                print s
            print 'data index: ', data_ind
            print '*********NOT-FOUND*********'
            for entity in data['stanford-entities']:
                print entity['surface-form']

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

print '%' * 150
s = 0
for a, b in knowledge_base.keys():
    s += knowledge_base[a, b]
print 'total relation extracted :', s

print 'number of independent organization', len(knowledge_base)
print 'number of final relation extracted', len(Knowledge_base)

print 'per_org pattern extracted: ', per_org_pattern_ext
print 'org_per pattern extracted: ', org_per_pattern_ext
