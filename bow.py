import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec
from math import log

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import LinearSVC
# build corpus
dataset = '20ng'

# shulffing
doc_name_list = []
doc_train_list = []
doc_test_list = []

f = open('data/' + dataset + '.txt', 'r')
for line in f.readlines():
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    if temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        doc_train_list.append(line.strip())
f.close()

doc_content_list = []
f = open('data/corpus/' + dataset + '.clean.txt', 'r')
for line in f.readlines():
    doc_content_list.append(line.strip())
f.close()

train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
print(train_ids)
random.shuffle(train_ids)

# partial labeled data


f = open('data/' + dataset + '.train.index', 'r')
lines = f.readlines()
f.close()
train_ids = [int(x.strip()) for x in lines]


#train_ids = train_ids[:int(0.2 * len(train_ids))]

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
print(test_ids)
random.shuffle(test_ids)

ids = train_ids + test_ids
print(ids)
print(len(ids))

train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size

shuffle_doc_name_list = []
shuffle_doc_words_list = []
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])

tfidf_vec = TfidfVectorizer() #max_features=50000
tfidf_matrix = tfidf_vec.fit_transform(shuffle_doc_words_list)
print(tfidf_matrix)
#tfidf_matrix_array = tfidf_matrix.toarray()

# BOW TFIDF + LR

#train_x = []
train_y = []

#test_x = []
test_y = []

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split(' ')

    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]

    if i < train_size:
        #train_x.append(tfidf_matrix_array[i])
        train_y.append(label)
    else:
        #test_x.append(tfidf_matrix_array[i])
        test_y.append(label)

#clf = svm.SVC(decision_function_shape='ovr', class_weight="balanced",kernel='linear')
#clf = LinearSVC(random_state=0)
clf = LogisticRegression(random_state=1)

clf.fit(tfidf_matrix[:train_size], train_y)
predict_y = clf.predict(tfidf_matrix[train_size:])

correct_count = 0
for i in range(len(test_y)):
    if predict_y[i] == test_y[i]:
        correct_count += 1

accuracy = correct_count * 1.0 / len(test_y)
print(dataset, accuracy)

print("Precision, Recall and F1-Score...")
print(metrics.classification_report(test_y, predict_y, digits=4))
