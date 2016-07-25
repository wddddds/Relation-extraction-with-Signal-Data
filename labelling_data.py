import json
import textwrap
import os


with open('/Users/kaka/Desktop/ML/msc_project/data/new_ceo/ceo_apr16.json') as gdata:
    data = gdata.readlines()
    data = data[612:1000]

with open('/Users/kaka/Desktop/ML/msc_project/data/labeled_data.json') as ldata:
    labeled_data = ldata.readlines()
    # print len(labeled_data)
    # print json.dumps(json.loads(labeled_data[1]), indent=4, separators=(',', ':'))
    # print json.loads(labeled_data[1])['exist_relation']

y = 'y'
n = 'n'

i = 611
for article in data:
    os.system('clear')
    print 'complete percentage: ', (1.0*i/1000)*100, "%  (", i, '/', '1000', ')'
    print '#'*100
    print '#'
    print '#'
    print '#'*100
    current_article = json.loads(article)
    current_article['article_index'] = i
    i += 1
    l = 0
    for s in textwrap.wrap(current_article['content'], width=80):
        print s
    index = 1
    exist_relation = 'ND'
    bad_entity = 'ND'
    Org_label = 'ND'
    old_ceo = 'ND'
    Old_CEO = 'ND'
    new_ceo = 'ND'
    New_CEO = 'ND'
    entities = (entity for entity in current_article['stanford-entities'] if entity['position'] == 'content')
    entities = list(entities)
    for entity in entities:
        print '*'*100
        print 'index of entity: ', index
        print '*'*50
        print json.dumps(entity, indent=4, separators=(',', ':'))
        index += 1
    while True:
        try:
            exist_relation = input('Is there a proper relation? input y or n:')
        except (SyntaxError, NameError):
            print 'not valid input, please try again...'
        if exist_relation not in ['y', 'n']:
            print 'not valid input, please try again...'
        else:
            break
    if exist_relation == 'y':
        while True:
            try:
                Org_label = input('index of organisation entity: ')
                Org_label = int(Org_label)
            except (NameError, SyntaxError):
                print 'please input the index of organisation entity: '
            except ValueError:
                print 'please input a valid number(index of organisation entity)'
            if Org_label > len(entities) or Org_label <= 0:
                print 'please input a valid number(index of organisation entity)'
            else:
                break

        # ask for new CEO information
        while True:
            try:
                new_ceo = input('is there new CEO? input y or n')
            except (NameError, SyntaxError):
                print 'please input y or n: '
            if exist_relation not in ['y', 'n']:
                print 'not valid input, please try again...'
            else:
                break
        if new_ceo == y:
            while True:
                try:
                    New_CEO = input('index of new CEO entity: ')
                    New_CEO = int(New_CEO)
                except (NameError, SyntaxError):
                    print 'please input y or n: '
                except ValueError:
                    print 'please input a valid number(index of new CEO entity)'
                if New_CEO > len(entities) or New_CEO <= 0:
                    print 'please input a valid number(index of new CEO entity)'
                else:
                    break
            New_CEO -= 1
            current_article['new_ceo'] = entities[New_CEO]
        else:
            current_article['no_new_ceo'] = 'y'

        # ask for Old CEO information
        while True:
            try:
                old_ceo = input('is there old CEO? input y or n')
            except (NameError, SyntaxError):
                print 'please input y or n: '
            if exist_relation not in ['y', 'n']:
                print 'not valid input, please try again...'
            else:
                break
        if old_ceo == y:
            while True:
                try:
                    Old_CEO = input('index of old CEO entity: ')
                    Old_CEO = int(Old_CEO)
                except (NameError, SyntaxError):
                    print 'please input y or n: '
                except ValueError:
                    print 'please input a valid number(index of old CEO entity)'
                if Old_CEO > len(entities) or Old_CEO <= 0:
                    print 'please input a valid number(index of old CEO entity)'
                else:
                    break
            Old_CEO -= 1
            current_article['old_ceo'] = entities[Old_CEO]
        else:
            current_article['no_old_ceo'] = 'y'
        current_article['exist_relation'] = 'y'
        Org_label -= 1
        current_article['org_appointed_new_ceo'] = entities[Org_label]
        with open('/Users/kaka/Desktop/ML/msc_project/data/labeled_data.json', 'a') as output_file:
            json.dump(current_article, output_file)
            output_file.write('\n')
    else:
        while True:
            try:
                bad_entity = input('Is there relations but named entity is not recognized? input y or n: ')
            except (NameError, SyntaxError):
                print 'not valid input, please try again...'
            if bad_entity not in ['y', 'n']:
                print 'not valid input, please try again...'
            else:
                break
        current_article['exist_relation'] = 'n'
        if bad_entity == y:
            current_article['bad_entity'] = 'y'
        with open('/Users/kaka/Desktop/ML/msc_project/data/labeled_data.json', 'a') as output_file:
            json.dump(current_article, output_file)
            output_file.write('\n')
