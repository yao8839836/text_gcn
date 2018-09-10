import wikipedia
import time
import pickle
from collections import Counter
import random

'''
t = time.time()

ny = wikipedia.page("Abdul%20Matlib%20Mazumdar")
print(ny.title)
print(ny.categories)
print(ny.content)
print(ny.links)

print("time=", "{:.5f}".format(time.time() - t))
'''

'''
f = open('data/long_abstracts_en.tql', 'r') # nq
wiki_lines = f.readlines()
f.close()

name_list = []
abstract_list = []
num_tokens = 0

for wiki_line in wiki_lines:
    if wiki_line[0] == '<':
        name_begin_index = wiki_line.find('resource/') + len('resource/')
        name_end_index= wiki_line.find('>')
        name = wiki_line[name_begin_index : name_end_index]
        print(name)
        name_list.append(name)
        abstract_begin_index = wiki_line.find('/abstract> "') + len('/abstract> "')
        # abstract>
        abstract_end_index = wiki_line.find('"@en <http://en.wikipedia.org')
        abstract = wiki_line[abstract_begin_index : abstract_end_index]
        print(abstract)
        abstract_list.append(abstract)
        num_tokens += len(abstract.split())
print(num_tokens)

name_str = '\n'.join(name_list)
f = open('data/wiki_long_abstracts_en_title.txt', 'w')
f.write(name_str)
f.close()

abstract_str = '\n'.join(abstract_list)
f = open('data/wiki_long_abstracts_en_text.txt', 'w')
f.write(abstract_str)
f.close()
'''

'''
f = open('data/instance_types_en.tql', 'r') 
wiki_category_lines = f.readlines()
f.close()

article_cat = {}
cat_article = {}

cnt = Counter()

article_cat_list = []

for wiki_line in wiki_category_lines:
    if wiki_line[0] == '<':
        name_begin_index = wiki_line.find('resource/') + len('resource/')
        name_end_index= wiki_line.find('>')
        name = wiki_line[name_begin_index : name_end_index]
        print(name)
        if wiki_line.find('<http://dbpedia.org/ontology/') != -1:

            category_begin_index = wiki_line.find('<http://dbpedia.org/ontology/') + len('<http://dbpedia.org/ontology/')
            category_end_index = wiki_line.find('> <http://en.wikipedia.org/wiki')
            category = wiki_line[category_begin_index : category_end_index]
            print(category)

        elif wiki_line.find('<http://www.w3.org/2002/07/owl#') != -1:

            category_begin_index = wiki_line.find('<http://www.w3.org/2002/07/owl#') + len('<http://www.w3.org/2002/07/owl#')
            category_end_index = wiki_line.find('> <http://en.wikipedia.org/wiki')
            category = wiki_line[category_begin_index : category_end_index]
            print(category)

        article_cat_list.append(name + '\t' + category)
        cnt[category] += 1

        if name not in article_cat:
            cat_set = set()
            cat_set.add(category)
            article_cat[name] = cat_set
        else:
            cat_set = article_cat[name]
            cat_set.add(category)          
            article_cat[name] = cat_set

        if category not in cat_article:
            article_set = set()
            article_set.add(name)
            cat_article[category] = article_set
        else:
            article_set = cat_article[category]
            article_set.add(name)
            cat_article[category] = article_set

print(cnt.most_common(100))

f = open("data/wiki_cate_counter.bin" ,'wb')
pickle.dump(cnt, f)
f.close()

f = open("data/article_cat.bin" ,'wb')
pickle.dump(article_cat, f)
f.close()

f = open("data/cat_article.bin" ,'wb')
pickle.dump(cat_article, f)
f.close()

article_cat_list_str = '\n'.join(article_cat_list)

f = open("data/wiki_article_actegory.txt", 'w')
f.write(article_cat_list_str)
f.close()
'''

'''
f = open("data/article_cat.bin", 'rb')
article_cat = pickle.load(f)
print(article_cat['Company'])
f.close()

count = 0
for article in article_cat:
    categories = article_cat[article]
    if len(categories) != 1:
        print(article, categories)
        count += 1
    else:
        #count += 1
        pass
print(count)

f = open("data/cat_article.bin", 'rb')
cat_article = pickle.load(f)
f.close()
print(len(cat_article))
# print(cat_article['Company'])

f = open("data/wiki_cate_counter.bin", 'rb')
cnt = pickle.load(f)
f.close()
print(cnt.most_common(100))

f = open("data/wiki_long_abstracts_en_title.txt", 'r')
title_lines = f.readlines()
f.close()
for i in range(len(title_lines)):
    title_lines[i] = title_lines[i].strip()

f = open("data/wiki_long_abstracts_en_text.txt", 'r')
content_lines = f.readlines()
f.close()
for i in range(len(content_lines)):
    content_lines[i] = content_lines[i].strip()

title_abstract_dict = dict(zip(title_lines, content_lines))


selected_list = ['Company', 'School', 'Album', 'Plant',
                 'Building', 'City', 'Athlete', 'Scientist', 'Village', 'Film']

selected_articles = []
content_list = []
for category in selected_list:
    articles = cat_article[category]
    
    for article in articles:
        categories = article_cat[article]
        categories = list(categories)
        content = ''
        if article in title_abstract_dict:
            content = title_abstract_dict[article]
        temp = content.split()
        if len(categories) == 1 and len(temp) >= 30:
            selected_articles.append(article + '\ttrain_or_test\t' + categories[0])
            content_list.append(content)

random_indexes = range(len(selected_articles))
random.shuffle(random_indexes)

random_indexes = random_indexes[:130000]

selected_articles = [selected_articles[j] for j in random_indexes]
content_list = [content_list[j] for j in random_indexes]

for i in range(len(selected_articles)):
    temp = str(selected_articles[i])
    if i < 120000:
        temp = temp.replace('train_or_test', 'train')
        selected_articles[i] = temp
    else:
        temp = temp.replace('train_or_test', 'test')
        selected_articles[i] = temp

selected_articles_str = '\n'.join(selected_articles)
f = open("data/wiki_selected_articles.txt", 'w')
f.write(selected_articles_str)
f.close()

selected_content_str = '\n'.join(content_list)
f = open("data/wiki_selected_articles_content.txt", 'w')
f.write(selected_content_str)
f.close()
'''

f = open("data/wiki.txt", 'r')
articles = f.readlines()
f.close()

new_split = []

for i in range(len(articles)):
    temp = articles[i].strip()
    print(temp)
    if i < 127000:
        temp = temp.split('\t')
        temp[1] = 'test'
    else:
        temp = temp.split('\t')
        temp[1] = 'train'
    temp = '\t'.join(temp)
    new_split.append(temp)

articles_str = '\n'.join(new_split)
f = open("data/wiki_1.txt", 'w')
f.write(articles_str)
f.close()