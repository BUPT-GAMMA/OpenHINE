import numpy as np

class DataProcess(object):
    def __init__(self, edgefile, data_type, link_type, relation_list):
        self.input_edge = edgefile
        self.data_type = data_type
        self.link_type = link_type
        self.relation_list = relation_list
        self.relation2id_dict = {}
        self.node2id_dict = {}
        self.matrix2id_dict = {}
        self.find_dict = {}
        self.node = {}
        self.adj_matrix = {}
    def load_node(self):
        for i in range(len(self.data_type)):
            self.node[self.data_type[i]] = set([])
        node_num = 0
        with open(self.input_edge) as file:
            file = file.readlines()
            for line in file:
                token = line.strip('\n').split("\t")
                source_type, target_type = token[2].split('-')
                source_id = self.data_type.find(source_type)
                target_id = self.data_type.find(target_type)
                self.node[self.data_type[source_id]].add(token[0])
                self.node[self.data_type[target_id]].add(token[1])
                node_num = node_num + 1
        return self.node

    def renum(self,output_fold):
        n2i_file = open((str(output_fold)+'node2id.txt'), 'w')
        idx1 = 0
        idx2 = 0
        num_node = 0
        for i in range(len(self.data_type)):
            num_node = num_node + len(self.node[self.data_type[i]])
        n2i_file.write(str(num_node))
        n2i_file.write('\n')
        for i in range(len(self.data_type)):
            idx2 = 0
            for j in self.node[self.data_type[i]]:
                tmpnode = self.data_type[i] + j
                n2i_file.write(tmpnode)
                n2i_file.write('\t' + str(idx1))
                n2i_file.write('\n')
                self.node2id_dict[tmpnode] = idx1
                self.matrix2id_dict[tmpnode] = idx2
                self.find_dict[self.data_type[i] + str(idx2)] = j
                idx1 = idx1 + 1
                idx2 = idx2 + 1
        #print(self.find_dict)
        r2i_file = open((str(output_fold)+'relation2id.txt'), 'w')
        self.relation_list = self.relation_list.split('+')
        #print(self.relation_list)
        num_relation = len(self.relation_list)
        r2i_file.write(str(num_relation))
        r2i_file.write('\n')
        for i, r in enumerate(self.relation_list):
            r = r.split('-')
            r = str(r[0])+str(r[1])
            r2i_file.write(str(r) + '\t')
            r2i_file.write(str(i) + '\n')
            self.relation2id_dict[r] = i
        return self.node2id_dict, self.matrix2id_dict, self.find_dict, self.relation2id_dict

    def load_matrix(self):
        link_type = self.link_type.split('+')
        with open(self.input_edge) as file:
            file = file.readlines()
            for i in link_type:
                #print(link_type)
                source_type, target_type = i.split('-')
                source_num = len(self.node[source_type])
                target_num = len(self.node[target_type])
                self.adj_matrix[i] = np.zeros([source_num, target_num], dtype=float)
            for line in file:

                token = line.strip('\n').split('\t')
                st, tt = token[2].split('-')
                row = self.matrix2id_dict[st+token[0]]
                col = self.matrix2id_dict[tt+token[1]]
                self.adj_matrix[token[2]][row][col] = token[3]
            #print(self.adj_matrix)
        return self.adj_matrix

    def generate_matrix(self,combination):
        combination = combination.split('|')
        for c in combination:
            source,target = c.split('==')
            source1, source2 = source.split('+')
            self.adj_matrix[target] = np.matmul(np.transpose(self.adj_matrix[source1]), self.adj_matrix[source2])
        return self.adj_matrix



