from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer

print(wn.synsets('dogs'))
print(wn.synsets('running'))

dog = wn.synset('dog.n.01')
print(dog.definition())

dog = wn.synset('run.n.05')
print(dog.definition())

dataset = 'ohsumed'

f = open('data/corpus/' + dataset + '_vocab.txt', 'r')
words = f.readlines()
f.close()

definitions = []

for word in words:
    word = word.strip()
    synsets = wn.synsets(word)
    word_defs = []
    for synset in synsets:
        syn_def = synset.definition()
        word_defs.append(syn_def)
    word_des = ' '.join(word_defs)
    if word_des == '':
        word_des = '<PAD>'
    definitions.append(word_des)

string = '\n'.join(definitions)


f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
f.write(string)
f.close()

tfidf_vec = TfidfVectorizer(max_features=50000)
tfidf_matrix = tfidf_vec.fit_transform(definitions)
tfidf_matrix_array = tfidf_matrix.toarray()
print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

word_vectors = []

for i in range(len(words)):
    word = words[i]
    vector = tfidf_matrix_array[i]
    str_vector = [] 
    for j in range(len(vector)):
        str_vector.append(str(vector[j]))
    temp = ' '.join(str_vector)
    word_vector = word + ' ' + temp
    word_vectors.append(word_vector)

string = '\n'.join(word_vectors)

f = open('data/corpus/' + dataset + '_word_vectors.txt', 'w')
f.write(string)
f.close()

