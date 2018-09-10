from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn import metrics
import numpy as np

dataset = 'mr'

# shulffing
doc_name_list = []
doc_train_list = []
doc_test_list = []

f = open('data/' + dataset + '_pvdm_200.vec', 'r') 
# _pvdm_200.vec
# _doc_vectors.txt
vector_lines = f.readlines()
f.close()

f = open('data/' + dataset + '.txt', 'r')
lines = f.readlines()
f.close()

train_x = []
train_y = []
test_x = []
test_y = []

for i in range(len(lines)):
    line = lines[i].strip()
    temp = line.split("\t")

    vector_line = vector_lines[i + 1].strip().split(' ') # +1
    doc_vec = vector_line[1:]
    for j in range(len(doc_vec)):
        doc_vec[j] = float(doc_vec[j])
    # print(doc_vec)
    # doc_vec = np.array(doc_vec)
    if temp[1].find('test') != -1:
        test_y.append(temp[2])
        test_x.append(doc_vec)
    elif temp[1].find('train') != -1:
        train_y.append(temp[2])
        train_x.append(doc_vec)

train_x = np.array(train_x)
test_x = np.array(test_x)

clf = LogisticRegression(random_state=0)
clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)

correct_count = 0
for i in range(len(test_y)):
    if predict_y[i] == test_y[i]:
        correct_count += 1

accuracy = correct_count * 1.0 / len(test_y)
print(dataset, accuracy)

print("Precision, Recall and F1-Score...")
print(metrics.classification_report(test_y, predict_y, digits=4))
