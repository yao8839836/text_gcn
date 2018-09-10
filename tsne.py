from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np

f = open('data/20ng.train.index', 'r')
lines = f.readlines()
f.close()
train_size = len(lines)


f = open('data/20ng_shuffle.txt', 'r')
lines = f.readlines()
f.close()

target_names = set()
labels = []
for line in lines:
    line = line.strip()
    temp = line.split('\t')
    labels.append(temp[2])
    target_names.add(temp[2])

target_names = list(target_names)

f = open('data/20ng_doc_vectors_1.txt', 'r')
lines = f.readlines()
f.close()

docs = []
for line in lines:
    temp = line.strip().split()
    values_str_list = temp[1:]
    values = [float(x) for x in values_str_list]
    docs.append(values)

fea = docs[train_size:]  # int(train_size * 0.9)
label = labels[train_size:]  # int(train_size * 0.9)
label = np.array(label)

fea = TSNE(n_components=2).fit_transform(fea)
pdf = PdfPages('20ng_gcn_doc_test_1.pdf')
cls = np.unique(label)

# cls=range(10)
fea_num = [fea[label == i] for i in cls]
for i, f in enumerate(fea_num):
    if cls[i] in range(10):
        plt.scatter(f[:, 0], f[:, 1], label=cls[i], marker='+')
    else:
        plt.scatter(f[:, 0], f[:, 1], label=cls[i])
# plt.legend(ncol=2,  )
# plt.legend(ncol=5,loc='upper center',bbox_to_anchor=(0.48, -0.08),fontsize=11)
# plt.ylim([-20,35])
# plt.title(md_file)
plt.tight_layout()
pdf.savefig()
plt.show()
pdf.close()
