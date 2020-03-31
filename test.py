import scipy.io as scio
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torch
import warnings
import argparse


class evaluation:
    def __init__(self, seed, label_file):
        self.seed = seed
        self.node_label = {}
        self.label = {}
        la = []
        with open(label_file) as f:
            for line in f:
                i, l = line.strip().split()
                self.label[i] = int(l)
                la.append(int(l))
        self.n_label = len(set(la))

    def evaluate_cluster(self, embedding_list):
        X = []
        Y = []
        for p in self.label:
            X.append(embedding_list[p])
            Y.append(self.label[p])

        Y_pred = KMeans(self.n_label, random_state=self.seed).fit(np.array(X)).predict(X)
        nmi = normalized_mutual_info_score(np.array(Y), Y_pred)
        ari = adjusted_rand_score(np.array(Y), Y_pred)
        return nmi, ari


    def evaluate_clf(self, embedding_list):
        X = []
        Y = []
        for p in self.label:
            X.append(embedding_list[p])
            Y.append(self.label[p])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=self.seed)

        LR = LogisticRegression()
        LR.fit(X_train, Y_train)
        Y_pred = LR.predict(X_test)

        micro_f1 = f1_score(Y_test, Y_pred, average='micro')
        macro_f1 = f1_score(Y_test, Y_pred, average='macro')
        return micro_f1, macro_f1


def str_list_to_float(str_list):
    return [float(item) for item in str_list]

def load_emd(filename):
    embedding_matrix = {}
    i = 0
    with open(filename) as infile:
        line = infile.readline().strip().split()
        n = int(line[0])
        for line in infile:
            emd = line.strip().split()
            if emd[1] != "nan":
                embedding_matrix[emd[0]] = str_list_to_float(emd[1:])
            else:
                print("nan error!")
            i = i + 1
        if(i != n):
            print("number of nodes error!")
    return embedding_matrix
parser = argparse.ArgumentParser(description="test")
parser.add_argument('-d', '--dataset', default='acm', type=str, help="Dataset")
parser.add_argument('-m', '--model', default='Metapath2vec', type=str, help='Train model')
parser.add_argument('-n', '--name', default='node.txt', type=str, help='Evaluation task')
parser.add_argument('-s', '--seed', default=0, type=str, help='seed')
args = parser.parse_args()

filename = "./output/embedding/" + args.model + "/" + args.name
label_file = "./dataset/" + args.dataset + "/label.txt"

_evaluation = evaluation(args.seed, label_file)
embedding_list = load_emd(filename)
NMI, ARI = _evaluation.evaluate_cluster(embedding_list)
micro, macro = _evaluation.evaluate_clf(embedding_list)



print('<Cluster>        NMI = %.4f, ARI = %.4f' % (NMI, ARI))

print('<Classification>     Micro-F1 = %.4f, Macro-F1 = %.4f' % (micro, macro))
# print('<Cluster> 		NMI = %.4f, ARI = %.4f(%.4f)' % (
#     np.mean(_NMI), np.mean(_ARI), np.std(_ARI, ddof=1)))

# print('<Classification> 	Micro-F1 = %.4f(%.4f), Macro-F1 = %.4f(%.4f)' % (
#     np.mean(_micro), np.std(_micro, ddof=1), np.mean(_macro), np.std(_macro, ddof=1)))
