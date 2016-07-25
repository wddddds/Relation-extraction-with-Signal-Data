import json
import numpy as np
import matplotlib.pyplot as plt
import utils
import operator

from collections import defaultdict, Counter


with open('/Users/kaka/Desktop/ML/msc_project/data/labeled_data500.json') as gdata:
    data = gdata.readlines()

relation = 0
no_relation = 0
bad_entity = 0
org_new_dist = []
org_old_dist = []
org_new_inside_content = []
org_old_inside_content = []
org_before_content = []
org_after_content = []
new_ceo_before_content = []
new_ceo_after_content = []
old_ceo_before_content = []
old_ceo_after_content = []

near_word = 3

for article in data:
    current_article = json.loads(article)
    if current_article['exist_relation'] == 'y':
        relation += 1
        org_start_offset = current_article['org_appointed_new_ceo']['start-offset']
        org_end_offset = current_article['org_appointed_new_ceo']['end-offset']
        if 'new_ceo' in current_article:
            new_ceo_start_offset = current_article['new_ceo']['start-offset']
            new_ceo_end_offset = current_article['new_ceo']['end-offset']

            try:
                ws = 0
                count = 1
                while ws <= near_word:
                    if current_article['content'][org_start_offset-count] == ' ':
                        ws += 1
                        count += 1
                    else:
                        count += 1
                org_before_content.append(current_article['content'][org_start_offset-count+1:org_start_offset])
            except IndexError:
                pass

            try:
                ws = 0
                count = 1
                while ws <= near_word:
                    if current_article['content'][org_end_offset+count] == ' ':
                        ws += 1
                        count += 1
                    else:
                        count += 1
                org_after_content.append(current_article['content'][org_end_offset:org_end_offset+count])
            except IndexError:
                pass

            try:
                ws = 0
                count = 1
                while ws <= near_word:
                    if current_article['content'][new_ceo_start_offset-count] == ' ':
                        ws += 1
                        count += 1
                    else:
                        count += 1
                new_ceo_before_content.append(current_article['content']
                                              [new_ceo_start_offset-count+1:new_ceo_start_offset])
            except IndexError:
                pass

            try:
                ws = 0
                count = 1
                while ws <= near_word:
                    if current_article['content'][new_ceo_end_offset+count] == ' ':
                        ws += 1
                        count += 1
                    else:
                        count += 1
                new_ceo_after_content.append(current_article['content'][new_ceo_end_offset:new_ceo_end_offset+count])
            except IndexError:
                pass

            if new_ceo_start_offset > org_end_offset:
                org_new_ceo_dist = new_ceo_end_offset - org_start_offset
                org_new_inside_content.append(current_article['content'][org_end_offset:new_ceo_start_offset])
            else:
                org_new_ceo_dist = org_end_offset - new_ceo_start_offset
                org_new_inside_content.append(current_article['content'][new_ceo_end_offset:org_start_offset])
            org_new_dist.append(org_new_ceo_dist)

        if 'old_ceo' in current_article:
            old_ceo_start_offset = current_article['old_ceo']['start-offset']
            old_ceo_end_offset = current_article['old_ceo']['end-offset']

            try:
                ws = 0
                count = 1
                while ws <= near_word:
                    if current_article['content'][old_ceo_start_offset-count] == ' ':
                        ws += 1
                        count += 1
                    else:
                        count += 1
                old_ceo_before_content.append(current_article['content']
                                              [old_ceo_start_offset-count+1:old_ceo_start_offset])
            except IndexError:
                pass

            try:
                ws = 0
                count = 1
                while ws <= near_word:
                    if current_article['content'][old_ceo_end_offset+count] == ' ':
                        ws += 1
                        count += 1
                    else:
                        count += 1
                old_ceo_after_content.append(current_article['content'][old_ceo_end_offset:old_ceo_end_offset+count])
            except IndexError:
                pass

            if old_ceo_start_offset > org_end_offset:
                org_old_ceo_dist = old_ceo_end_offset - org_start_offset
                org_old_inside_content.append(current_article['content'][org_end_offset:old_ceo_start_offset])
            else:
                org_old_ceo_dist = org_end_offset - old_ceo_start_offset
                org_old_inside_content.append(current_article['content'][old_ceo_end_offset:org_start_offset])
            org_old_dist.append(org_old_ceo_dist)

    try:
        if current_article['bad_entity'] == 'y':
            bad_entity += 1
    except KeyError:
        pass

print 'amount of labeled articles: ', len(data)
print 'total ground truth relations: ', relation
print 'amount of relations that cant be extract due to bad entity:', bad_entity

'''
===================================================================================
This part we explore the distance between organization entity and new CEO entity
===================================================================================
'''

org_new_dist_array = np.asarray(org_new_dist)
org_old_dist_array = np.asarray(org_old_dist)

print 'mean distance between organisation and new CEO: ', np.mean(org_new_dist_array)
print 'mean distance between organisation and old CEO: ', np.mean(org_old_dist_array)


# distance histogram plot
plt.hist(org_new_dist_array, bins=100)
plt.title("org new CEO distance")
plt.xlabel("distance")
plt.ylabel("Frequency")
plt.show()

plt.hist(org_old_dist_array, bins=100)
plt.title("org old CEO distance")
plt.xlabel("distance")
plt.ylabel("Frequency")
plt.show()

'''
===================================================================================
This part we explore the content between organization entity and new CEO entity,
including
===================================================================================
'''

# content between entities statistics
vocabulary_org_new = defaultdict(lambda: 0)
for s in org_new_inside_content:
    words = utils.tokenize(s)
    for word in words:
        vocabulary_org_new[word] += 1

vocabulary_org_old = defaultdict(lambda: 0)
for s in org_old_inside_content:
    words = utils.tokenize(s)
    for word in words:
        vocabulary_org_old[word] += 1

vocabulary_org_new_sorted = sorted(vocabulary_org_new.items(), key=operator.itemgetter(1), reverse=True)
vocabulary_org_old_sorted = sorted(vocabulary_org_old.items(), key=operator.itemgetter(1), reverse=True)

print vocabulary_org_new_sorted
print vocabulary_org_old_sorted

org_new_dict = {}
show_amount = 0
for w, v in vocabulary_org_new_sorted:
    if show_amount < 20:
        org_new_dict[w] = int(v)
    show_amount += 1

org_old_dict = {}
show_amount = 0
for w, v in vocabulary_org_old_sorted:
    if show_amount < 20:
        org_old_dict[w] = int(v)
    show_amount += 1

c = Counter(org_new_dict).items()
c.sort(key=operator.itemgetter(1), reverse=True)
labels, values = zip(*c)
plt.xlim([-1, len(org_new_dict)])
plt.title("frequent words between ORG entity and new ceo entity")
plt.bar(range(len(org_new_dict)), values, align='center', width=0.9)
plt.xticks(range(len(org_new_dict)), labels, rotation=45)

plt.show()

c = Counter(org_old_dict).items()
c.sort(key=operator.itemgetter(1), reverse=True)
labels, values = zip(*c)
plt.xlim([-1, len(org_old_dict)])
plt.title("frequent words between ORG entity and old ceo entity")
plt.bar(range(len(org_old_dict)), values, align='center', width=0.9)
plt.xticks(range(len(org_old_dict)), labels, rotation=45)

plt.show()

'''
===================================================================================
This part we explore the nearest words with ORG entity
===================================================================================
'''
# content between entities statistics
vocabulary_org_before = defaultdict(lambda: 0)
for s in org_before_content:
    words = utils.tokenize(s)
    for word in words:
        vocabulary_org_before[word] += 1

vocabulary_org_after = defaultdict(lambda: 0)
for s in org_after_content:
    words = utils.tokenize(s)
    for word in words:
        vocabulary_org_after[word] += 1

vocabulary_org_before_sorted = sorted(vocabulary_org_before.items(), key=operator.itemgetter(1), reverse=True)
vocabulary_org_after_sorted = sorted(vocabulary_org_after.items(), key=operator.itemgetter(1), reverse=True)

print vocabulary_org_before_sorted
print vocabulary_org_after_sorted

org_before_dict = {}
show_amount = 0
for w, v in vocabulary_org_before_sorted:
    if show_amount < 20:
        org_before_dict[w] = int(v)
    show_amount += 1

org_after_dict = {}
show_amount = 0
for w, v in vocabulary_org_after_sorted:
    if show_amount < 20:
        org_after_dict[w] = int(v)
    show_amount += 1

c = Counter(org_before_dict).items()
c.sort(key=operator.itemgetter(1), reverse=True)
labels, values = zip(*c)
plt.xlim([-1, len(org_before_dict)])
plt.title("frequent words before ORG entity")
plt.bar(range(len(org_before_dict)), values, align='center', width=0.9)
plt.xticks(range(len(org_before_dict)), labels, rotation=45)

plt.show()

c = Counter(org_after_dict).items()
c.sort(key=operator.itemgetter(1), reverse=True)
labels, values = zip(*c)
plt.xlim([-1, len(org_after_dict)])
plt.title("frequent words after ORG entity")
plt.bar(range(len(org_after_dict)), values, align='center', width=0.9)
plt.xticks(range(len(org_after_dict)), labels, rotation=45)

plt.show()

'''
===================================================================================
This part we explore the nearest words with new CEO entity
===================================================================================
'''

# content between entities statistics
vocabulary_new_ceo_before = defaultdict(lambda: 0)
for s in new_ceo_before_content:
    words = utils.tokenize(s)
    for word in words:
        vocabulary_new_ceo_before[word] += 1

vocabulary_new_ceo_after = defaultdict(lambda: 0)
for s in new_ceo_after_content:
    words = utils.tokenize(s)
    for word in words:
        vocabulary_new_ceo_after[word] += 1

vocabulary_new_ceo_before_sorted = sorted(vocabulary_new_ceo_before.items(), key=operator.itemgetter(1), reverse=True)
vocabulary_new_ceo_after_sorted = sorted(vocabulary_new_ceo_after.items(), key=operator.itemgetter(1), reverse=True)

print vocabulary_new_ceo_before_sorted
print vocabulary_new_ceo_after_sorted

new_ceo_before_dict = {}
show_amount = 0
for w, v in vocabulary_new_ceo_before_sorted:
    if show_amount < 20:
        new_ceo_before_dict[w] = int(v)
    show_amount += 1

new_ceo_after_dict = {}
show_amount = 0
for w, v in vocabulary_new_ceo_after_sorted:
    if show_amount < 20:
        new_ceo_after_dict[w] = int(v)
    show_amount += 1

c = Counter(new_ceo_before_dict).items()
c.sort(key=operator.itemgetter(1), reverse=True)
labels, values = zip(*c)
plt.xlim([-1, len(new_ceo_before_dict)])
plt.title("frequent words before new CEO entity")
plt.bar(range(len(new_ceo_before_dict)), values, align='center', width=0.9)
plt.xticks(range(len(new_ceo_before_dict)), labels, rotation=45)

plt.show()

c = Counter(new_ceo_after_dict).items()
c.sort(key=operator.itemgetter(1), reverse=True)
labels, values = zip(*c)
plt.xlim([-1, len(new_ceo_after_dict)])
plt.title("frequent words after new CEO entity")
plt.bar(range(len(new_ceo_after_dict)), values, align='center', width=0.9)
plt.xticks(range(len(new_ceo_after_dict)), labels, rotation=45)

plt.show()

'''
===================================================================================
This part we explore the nearest words with old CEO entity
===================================================================================
'''

# content between entities statistics
vocabulary_old_ceo_before = defaultdict(lambda: 0)
for s in old_ceo_before_content:
    words = utils.tokenize(s)
    for word in words:
        vocabulary_old_ceo_before[word] += 1

vocabulary_old_ceo_after = defaultdict(lambda: 0)
for s in old_ceo_after_content:
    words = utils.tokenize(s)
    for word in words:
        vocabulary_old_ceo_after[word] += 1

vocabulary_old_ceo_before_sorted = sorted(vocabulary_old_ceo_before.items(), key=operator.itemgetter(1), reverse=True)
vocabulary_old_ceo_after_sorted = sorted(vocabulary_old_ceo_after.items(), key=operator.itemgetter(1), reverse=True)

print vocabulary_old_ceo_before_sorted
print vocabulary_old_ceo_after_sorted

old_ceo_before_dict = {}
show_amount = 0
for w, v in vocabulary_old_ceo_before_sorted:
    if show_amount < 20:
        old_ceo_before_dict[w] = int(v)
    show_amount += 1

old_ceo_after_dict = {}
show_amount = 0
for w, v in vocabulary_old_ceo_after_sorted:
    if show_amount < 20:
        old_ceo_after_dict[w] = int(v)
    show_amount += 1

c = Counter(old_ceo_before_dict).items()
c.sort(key=operator.itemgetter(1), reverse=True)
labels, values = zip(*c)
plt.xlim([-1, len(old_ceo_before_dict)])
plt.title("frequent words before old CEO entity")
plt.bar(range(len(old_ceo_before_dict)), values, align='center', width=0.9)
plt.xticks(range(len(old_ceo_before_dict)), labels, rotation=45)

plt.show()

c = Counter(old_ceo_after_dict).items()
c.sort(key=operator.itemgetter(1), reverse=True)
labels, values = zip(*c)
plt.xlim([-1, len(old_ceo_after_dict)])
plt.title("frequent words after old CEO entity")
plt.bar(range(len(old_ceo_after_dict)), values, align='center', width=0.9)
plt.xticks(range(len(old_ceo_after_dict)), labels, rotation=45)

plt.show()

'''
===================================================================================
This part we explore how many relations occurs in a single sentence
===================================================================================
'''
count = 0
for relation in org_new_inside_content:
    if '.' not in relation:
        count += 1
    else:
        pass

print 'among', len(org_new_inside_content), 'new CEO relations,', count, ' of them occur in one sentence'
