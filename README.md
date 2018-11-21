# text_gcn

The implementation of Text GCN in our paper:

Liang Yao, Chengsheng Mao, Yuan Luo. "Graph Convolutional Networks for Text Classification." In 33rd AAAI Conference on Artificial Intelligence (AAAI-19)


# Require

Python 2.7 or 3.6

Tensorflow >= 1.4.0

# Reproduing Results

1. Run `python remove_words.py 20ng`

2. Run `python build_graph.py 20ng`

3. Run `python train.py 20ng`

4. Change `20ng` in above 3 command lines to `R8`, `R52`, `ohsumed` and `mr` when producing results for other datasets.

# Example input data

1. `/data/20ng.txt` indicates document names, training/test split, document labels. Each line is for a document.

2. `/data/corpus/20ng.txt` contains raw text of each document, each line is for the corresponding line in `/data/20ng.txt`