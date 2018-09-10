import re
# build corpus


dataset = '20ng'

f = open('data/' + dataset + '.txt', 'r')
lines = f.readlines()
docs = []
for line in lines:
    temp = line.split("\t")
    doc_file = open(temp[0], 'r')
    doc_content = doc_file.read()
    doc_file.close()
    print(temp[0], doc_content)
    doc_content = doc_content.replace('\n', ' ')
    docs.append(doc_content)


corpus_str = '\n'.join(docs)
f.close()

f = open('data/corpus/' + dataset + '.txt', 'w')
f.write(corpus_str)
f.close()


'''
# datasets from PTE paper
f = open('data/dblp/label_train.txt', 'r')
lines = f.readlines()
f.close()

doc_id = 0
doc_name_list = []
for line in lines:
    string = str(doc_id) + '\t' + 'train' + '\t' + line.strip()
    doc_name_list.append(string)
    doc_id += 1

f = open('data/dblp/label_test.txt', 'r')
lines = f.readlines()
f.close()

for line in lines:
    string = str(doc_id) + '\t' + 'test' + '\t' + line.strip()
    doc_name_list.append(string)
    doc_id += 1

doc_list_str = '\n'.join(doc_name_list)

f = open('data/dblp.txt', 'w')
f.write(doc_list_str)
f.close()

# TREC, R8, R52, WebKB

dataset = 'R52'

f = open('data/' + dataset + '/train.txt', 'r')
lines = f.readlines()
f.close()

doc_id = 0
doc_name_list = []
doc_content_list = []

for line in lines:
    line = line.strip()
    label = line[:line.find('\t')]
    content = line[line.find('\t') + 1:]
    string = str(doc_id) + '\t' + 'train' + '\t' + label
    doc_name_list.append(string)
    doc_content_list.append(content)
    doc_id += 1

f = open('data/' + dataset + '/test.txt', 'r')
lines = f.readlines()
f.close()

for line in lines:
    line = line.strip()
    label = line[:line.find('\t')]
    content = line[line.find('\t') + 1:]
    string = str(doc_id) + '\t' + 'test' + '\t' + label
    doc_name_list.append(string)
    doc_content_list.append(content)
    doc_id += 1

doc_list_str = '\n'.join(doc_name_list)

f = open('data/' + dataset + '.txt', 'w')
f.write(doc_list_str)
f.close()

doc_name_list_str = '\n'.join(doc_name_list)

f = open('data/' + dataset + '.txt', 'w')
f.write(doc_list_str)
f.close()

doc_content_list_str = '\n'.join(doc_content_list)

f = open('data/corpus/' + dataset + '.txt', 'w')
f.write(doc_content_list_str)
f.close()
'''
