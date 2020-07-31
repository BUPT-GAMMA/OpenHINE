import numpy as np
from scipy.sparse import csr_matrix
from src.utils.utils import str_list_to_float

def inverse_relation(s):
    return s[2] + s[1] + s[0]

def str_int(elem):
    return int(elem)


class HIN(object):
    def __init__(self, inputfold, data_type, relation_list):
        self.input_edge = inputfold + 'edge.txt'
        self.label_file = inputfold + 'label.txt'
        self.fea_file = inputfold + 'feature.txt'
        self.data_type = data_type
        self.relation_list = relation_list.split('+')
        self.relation2id_dict = {}
        self.node2id_dict = {}
        self.matrix2id_dict = {}
        self.find_dict = {}
        self.edge_weight = {}
        self.node = self.load_node()
        self.adj_matrix = {}
        self.relation_dict = self.load_relation()
        self.feature = {}


    def load_node(self):
        node = {}
        for i in range(len(self.data_type)):
            node[self.data_type[i]] = set([])
        node_num = 0
        with open(self.input_edge) as file:
            file = file.readlines()
            for line in file:
                token = line.strip('\n').split("\t")
                source_type, target_type = token[2].split('-')
                source_id = self.data_type.find(source_type)
                target_id = self.data_type.find(target_type)
                node[self.data_type[source_id]].add(token[0])
                node[self.data_type[target_id]].add(token[1])
                node_num = node_num + 1

        for i in node:
            node[i] = sorted(list(node[i]), key=str_int)
        idx1 = 0
        for i in self.data_type:
            idx2 = 0
            for j in node[i]:
                tmpnode = i + j
                self.node2id_dict[tmpnode] = idx1
                self.matrix2id_dict[tmpnode] = idx2
                self.find_dict[i + str(idx2)] = j
                idx1 = idx1 + 1
                idx2 = idx2 + 1
        return node

    def load_relation(self):
        relation_dict = dict()
        for relation in self.relation_list:
            relation_dict[relation] = dict()
            inv_relation = inverse_relation(relation)
            relation_dict[inv_relation] = dict()
        with open(self.input_edge) as file:
            for line in file:
                token = line.strip().split()
                src, tar = token[0], token[1]
                relation = token[2]
                if src not in relation_dict[relation]:
                    relation_dict[relation][src] = []
                relation_dict[relation][src].append(tar)

                # inv_relation = inverse_relation(relation)
                # if tar not in relation_dict[inv_relation]:
                #     relation_dict[inv_relation][tar] = []
                # relation_dict[inv_relation][tar].append(src)
                if relation not in self.edge_weight:
                    self.edge_weight[relation] = {}
                self.edge_weight[relation][str(src)+'-'+str(tar)] = token[3]

        for i, r in enumerate(self.relation_list):
            self.relation2id_dict[r] = i
        return relation_dict

    def renum(self, output_fold):

        for i, r in enumerate(self.relation_list):
            self.relation2id_dict[r] = i
        return self.node2id_dict, self.matrix2id_dict, self.find_dict, self.relation2id_dict

    def load_matrix(self):
        for relation in self.relation_dict:
            source_type, target_type = relation.split('-')
            n1 = len(self.node[source_type])
            n2 = len(self.node[target_type])
            self.adj_matrix[relation] = self.re2mtx(relation, n1, n2)
            # print(self.adj_matrix)

    def re2mtx(self, relation, n1, n2):
        row = []
        col = []
        source_type, target_type = relation.split('-')
        for i in self.relation_dict[relation]:
            for j in self.relation_dict[relation][i]:
                row.append(self.matrix2id_dict[source_type + i])
                col.append(self.matrix2id_dict[target_type + j])
        data = np.ones(len(row))
        return csr_matrix((data, (row, col)), shape=(n1, n2))

    def generate_matrix(self, combination):
        combination = combination.split('|')
        for c in combination:
            source, target = c.split('==')
            source1, source2 = source.split('+')
            self.adj_matrix[target] = self.adj_matrix[source1] * self.adj_matrix[source2]

    def node_type_mapping(self, output_data):
        with open(output_data, "w") as f:
            for type in self.node:
                for i in self.node[type]:
                    outline = type + i + ' ' + type + "\n"
                    f.write(outline)

    def load_label(self):
        label = {}
        set_label = []
        with open(self.label_file) as file:
            for i in file:
                line = i.strip().split()
                label[line[0]] = int(line[1])
                set_label.append(int(line[1]))
        return label, len(set(set_label))

    def load_fea(self):
        try:
            with open(self.fea_file) as infile:
                for line in infile.readlines()[1:]:
                    emd = line.strip().split()
                    self.feature[emd[0]] = np.array(str_list_to_float(emd[1:]))
        except FileNotFoundError:
            print("The dataset directory can't find the feature file!")
            exit(1)

