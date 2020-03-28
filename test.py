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


def evaluate_cluster(embedding_matrix, n_label):
    embedding_list = embedding_matrix.tolist()

    X = []
    Y = []
    for p in label:
        X.append(embedding_list[p])
        Y.append(label[p])

    Y_pred = KMeans(n_label, random_state=0).fit(np.array(X)).predict(X)
    nmi = normalized_mutual_info_score(np.array(Y), Y_pred)
    ari = adjusted_rand_score(np.array(Y), Y_pred)
    return nmi, ari


def evaluate_clf(embedding_matrix, seed):
    embedding_list = embedding_matrix.tolist()

    X = []
    Y = []
    for p in label:
        X.append(embedding_list[p])
        Y.append(label[p])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)

    LR = LogisticRegression()
    LR.fit(X_train, Y_train)
    Y_pred = LR.predict(X_test)

    micro_f1 = f1_score(Y_test, Y_pred, average='micro')
    macro_f1 = f1_score(Y_test, Y_pred, average='macro')
    return micro_f1, macro_f1


def str_list_to_float(str_list):
    return [float(item) for item in str_list]


parser = argparse.ArgumentParser(description="test")
parser.add_argument('-d', '--dataset', default='acm', type=str, help="Dataset")
parser.add_argument('-m', '--model', default='PME', type=str, help='Train model')
parser.add_argument('-n', '--name', default='node.txt', type=str, help='Evaluation task')
parser.add_argument('-s', '--seed', default=0, type=str, help='seed')
args = parser.parse_args()

if args.dataset == "dblp":
    s = "a"
    n_label = 4
else:
    s = "p"
    n_label = 3
dataset = "acm"
filename = "./output/embedding/" + args.model + "/" + args.name
label = {}
label_file = "./dataset/" + args.dataset + "/label.txt"
with open(label_file) as f:
    for line in f:
        i, l = line.strip().split()
        label[int(i[1:])] = int(l)

i = -1
with open(filename) as infile:
    line = infile.readline().strip().split()
    embedding_matrix = np.zeros((int(line[0]), int(line[1])))
    for line in infile:
        emd = line.strip().split()
        if emd[0][0] == s:
            if emd[1] != "nan":
                embedding_matrix[int(emd[0][1:]), :] = str_list_to_float(emd[1:])
# _NMI = []
# _ARI = []
# _micro = []
# _macro = []
# for i in range(1, 2):
NMI, ARI = evaluate_cluster(embedding_matrix, n_label)
# _NMI.append(NMI)
# _ARI.append(ARI)

micro, macro = evaluate_clf(embedding_matrix, args.seed)
# _micro.append(micro)
# _macro.append(macro)

# auc, f1, acc = evaluate_lp(lp_embedding)
print('<Cluster>        NMI = %.4f, ARI = %.4f' % (
    NMI, ARI))

print('<Classification>     Micro-F1 = %.4f, Macro-F1 = %.4f' % (
    micro, macro))
# print('<Cluster> 		NMI = %.4f, ARI = %.4f(%.4f)' % (
#     np.mean(_NMI), np.mean(_ARI), np.std(_ARI, ddof=1)))

# print('<Classification> 	Micro-F1 = %.4f(%.4f), Macro-F1 = %.4f(%.4f)' % (
#     np.mean(_micro), np.std(_micro, ddof=1), np.mean(_macro), np.std(_macro, ddof=1)))
