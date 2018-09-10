from sklearn.datasets import fetch_rcv1
rcv1 = fetch_rcv1()
print(rcv1.data.shape)
print(rcv1.target.shape)
print(rcv1.sample_id[:3])

target_names_list = rcv1.target_names.tolist()
sample_id_list = rcv1.sample_id.tolist()
print(target_names_list)

non_leaf = set()
f = open('data/rcv1/rcv1.topics.hier.expanded.txt', 'r')
lines = f.readlines()
for line in lines:
    temp = line.strip().split()
    label = temp[1]
    print(label)
    non_leaf.add(label)

leaf = []
for target_name in target_names_list:
    if target_name not in non_leaf:
        leaf.append(target_name)

print(leaf, len(leaf))

leaf_idx = []
for l in leaf:
    idx = target_names_list.index(l)
    leaf_idx.append(idx)
print(leaf_idx)

select_ids = []
leaf_set = set(leaf_idx)
count = 0
for i in range(rcv1.data.shape[0]):
    
    categories = rcv1.target[i].nonzero()[1]
    cat_count = 0
    cat_i = -2
    for category in categories:
        if category in leaf_set:
            cat_count += 1
            cat_i = category
    if cat_count == 1:
        count += 1
        doc_id = sample_id_list[i]
        if doc_id > 26150:
            select_ids.append(str(doc_id) + '\ttest\t' + target_names_list[cat_i])
        else:
            select_ids.append(str(doc_id) + '\ttrain\t' + target_names_list[cat_i])
        
print(count)
print(rcv1.sample_id)

select_ids_str = '\n'.join(select_ids)
f = open('data/rcv1/rcv1.txt', 'w')
f.write(select_ids_str)
f.close()

f = open('data/rcv1/lyrl2004_tokens_train.dat', 'r')
train_str = f.read()
f.close()
f = open('data/rcv1/lyrl2004_tokens_test_pt0.dat', 'r')
test_str0 = f.read()
f.close()
f = open('data/rcv1/lyrl2004_tokens_test_pt1.dat', 'r')
test_str1 = f.read()
f.close()
f = open('data/rcv1/lyrl2004_tokens_test_pt2.dat', 'r')
test_str2 = f.read()
f.close()
f = open('data/rcv1/lyrl2004_tokens_test_pt3.dat', 'r')
test_str3 = f.read()
f.close()

all_str = train_str + test_str0 + test_str1 + test_str2 + test_str3

docs = all_str.split('\n\n')

doc_id_words_map = {}
for doc in docs:
    doc_lines = doc.split('\n')
    if len(doc_lines) >= 2:
        temp = doc_lines[0].split()
        doc_id = temp[1]
        doc_words_lines = doc_lines[2:]
        doc_words = ' '.join(doc_words_lines)
        doc_id_words_map[doc_id] = doc_words

docs_content = []
for line in select_ids:
    temp = line.split()
    doc_id = temp[0]
    words = doc_id_words_map[doc_id]
    docs_content.append(words)

docs_content_str = '\n'.join(docs_content)
f = open('data/rcv1/rcv1_content.txt', 'w')
f.write(docs_content_str)
f.close()