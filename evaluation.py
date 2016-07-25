import json
import utils
import re
import textwrap

from collections import defaultdict

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

print '\n', '%' * 150
s = 0
for a, b in knowledge_base.keys():
    s += knowledge_base[a, b]
print 'total relation extracted :', s

print 'number of independent organization', len(knowledge_base)
print 'number of final relation extracted', len(Knowledge_base)

print 'per_org pattern extracted: ', per_org_pattern_ext
print 'org_per pattern extracted: ', org_per_pattern_ext

print correct, wrong
precision = 1.0*correct/s
recall = 1.0*correct/144
print 'precision: ', precision
print 'recall: ', recall
print 'F1-score: ', 2*(precision * recall)/(precision + recall)
