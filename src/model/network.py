#!/usr/bin/python
# -*- encoding: utf8 -*-

import random
import numpy as np


__author__ = 'sheep'


class HIN(object):
    '''
        a heterogeneous information network
        which support multigraph, weighted edges
    '''
    def __init__(self):
        self.graph = {} #{from_id: {to_id: {edge_class_id: weight}}}
        #TODO correct the name
        self.class_nodes = {} #{node_class: set([node_id])}
        self.edge_class2id = {} #{edge_class: edge_class_id}
        self.node2id = {} #{node: node_id}
        self.k_hop_neighbors = {} #{k: {id_: set(to_ids)}}
        self.edge_class_id_available_node_class = {} # {edge_class_id:
                                                     #  (from_node_class, to_node_class)}

    def __eq__(self, other):
        if not isinstance(other, HIN):
            return False
        if self.graph != other.graph:
            return False
        if self.class_nodes != other.class_nodes:
            return False
        if self.edge_class2id != other.edge_class2id:
            return False
        if self.node2id != other.node2id:
            return False
        if self.edge_class_id_available_node_class != other.edge_class_id_available_node_class:
            return False
        return True

    def node_count(self):
        return len(self.graph)

    def edge_count(self):
        count = 0
        for from_id in self.graph:
            for to_ids in self.graph[from_id].values():
                count += len(to_ids)
        return count

    def dump_to_file(self, fname):
        with open(fname, 'w') as f:
            cPickle.dump(self, f)

    @staticmethod
    def load_from_file(fname):
        with open(fname, 'r') as f:
            return cPickle.load(f)

    @staticmethod
    def load_from_edge_file(fname):
        g = HIN()
        with open(fname, 'r') as f:
            for line in f:
                line = line.strip()
                tokens = line.split()
                g.add_edge(tokens[0], '', tokens[1], '', '',
                           weight=int(tokens[2]))
                g.add_edge(tokens[1], '', tokens[0], '', '',
                           weight=int(tokens[2]))
        return g

    @staticmethod
    def get_inverse_edge(edge):
        inversed = edge[::-1]
        if '>' in inversed:
            inversed = inversed.replace('>', '<')
        elif '<' in inversed:
            inversed = inversed.replace('<', '>')
        return inversed

    def get_edge_class_inverse_mappling(self):
        inverse_mapping = {}
        for edge_class in self.edge_class2id.keys():
            inversed = edge_class[::-1]
            if '>' in inversed:
                inversed = inversed.replace('>', '<')
            elif '<' in inversed:
                inversed = inversed.replace('<', '>')
            if inversed != edge_class and inversed in self.edge_class2id:
                inverse_mapping[str(self.edge_class2id[edge_class])] = str(self.edge_class2id[inversed])
        return inverse_mapping

    def add_edge(self, from_node, from_class, to_node, to_class,edge_class,
                 weight = 1):
        if edge_class not in self.edge_class2id:
            self.edge_class2id[edge_class] = len(self.edge_class2id)
        edge_id = self.edge_class2id[edge_class]

        if edge_id not in self.edge_class_id_available_node_class:
            self.edge_class_id_available_node_class[edge_id] = (from_class, to_class)

        if from_node not in self.node2id:
            self.node2id[from_node] = len(self.node2id)
        from_id = self.node2id[from_node]
        if to_node not in self.node2id:
            self.node2id[to_node] = len(self.node2id)
        to_id = self.node2id[to_node]

        if from_class not in self.class_nodes:
            self.class_nodes[from_class] = set()
        self.class_nodes[from_class].add(from_id)

        if to_class not in self.class_nodes:
            self.class_nodes[to_class] = set()
        self.class_nodes[to_class].add(to_id)

        if from_id not in self.graph:
            self.graph[from_id] = {to_id: {edge_id: weight}}
            return
        if to_id not in self.graph[from_id]:
            self.graph[from_id][to_id] = {edge_id: weight}
            return
        self.graph[from_id][to_id][edge_id] = weight

    def has_node(self, node):
        return node in self.node2id

    def has_edge(self, from_node, to_node, edge_class=None):
        if from_node not in self.node2id:
            return False
        if to_node not in self.node2id:
            return False
        if edge_class is not None:
            edge_class_id = self.edge_class2id[edge_class]
            from_id = self.node2id[from_node]
            to_id = self.node2id[to_node]
            #TODO graph form is {from_id: {to_id: {edge_class_id: weight}}} below mistake?
            #ORIGINAL:
            #if to_id in self.graph[from_id][edge_class_id]:
            #    return True
            #return False
            #MODIFIED
            if to_id in self.graph[from_id]:
                if edge_class_id in self.graph[from_id][to_id]:
                    return True
                return False
            return False
        else:
            from_id = self.node2id[from_node]
            to_id = self.node2id[to_node]
            for to_ids in self.graph[from_id].values():
                if to_id in to_ids:
                    return True
            return False

    def common_neighbors(self, from_id, to_id):
        from_neighbors = set(self.graph[from_id][0].keys())
        to_neighbors = set(self.graph[to_id][0].keys())
        return from_neighbors.intersection(to_neighbors)

    def ratio_common_neighbors(self, from_id, to_id):
        from_neighbors = set(self.graph[from_id][0].keys())
        to_neighbors = set(self.graph[to_id][0].keys())
        intersection = from_neighbors.intersection(to_neighbors)
        union = from_neighbors.union(to_neighbors)
        return float(len(intersection))/len(union)

    def print_statistics(self):
        for c, nodes in self.class_nodes.items():
            print(c, len(nodes))
        class_count = {}
        for class_edges in self.graph.values():
            for class_, to_ids in class_edges.items():
                if class_ not in class_count:
                    class_count[class_] = len(to_ids)
                    continue
                class_count[class_] += len(to_ids)
        print(self.edge_class2id)
        for class_, count in class_count.items():
            print(class_, count)

    def to_homogeneous_network(self):
        aset = set()
        for nodes in self.class_nodes.values():
            aset.update(nodes)
        self.class_nodes = {'': aset}

        self.edge_class2id = {'': 0}

        self.edge_class_id_available_node_class = {}

        graph = {}
        for from_id, to_edges in self.graph.items():
            graph[from_id] = {}
            for to_id, edges in to_edges.items():
                weight = sum(edges.values())
                graph[from_id][to_id] = {0: weight}
        self.graph = graph

    #TODO speed up
    def to_weighted_edge_list(self, with_edge_class_id=False):
        '''
            ignore edge type
        '''
        id2node = dict([(v, k) for k, v in self.node2id.items()])
        edges = []
        for node_id in self.graph:
            for to_id, to_edges in self.graph[node_id].items():
                for edge_class_id, weight in to_edges.items():
                    if with_edge_class_id:
                        edge = (id2node[node_id],
                                id2node[to_id],
                                weight,
                                edge_class_id)
                    else:
                        edge = (id2node[node_id], id2node[to_id], weight)
                    edges.append(edge)
        return edges

    def dump_edge_list_file(self, fname):
        node2class = {}
        for c, ids in self.class_nodes.items():
            for id_ in ids:
                node2class[id_] = c

        edge_class_id2edge_class = {}
        for c, id_ in self.edge_class2id.items():
            edge_class_id2edge_class[id_] = c

        with open(fname, 'w') as f:
            for node_id in self.graph:
                for edge_class_id, tos in self.graph[node_id].items():
                    for to_id, weight in tos.items():
                        line = ('%d\t%d\t%f\t%s\t%s\t%s\n'
                                '' % (node_id,
                                  to_id,
                                  weight,
                                  node2class[node_id],
                                  node2class[to_id],
                                  edge_class_id2edge_class[edge_class_id]))
                        f.write(line)

    def get_ids(self):
        return sorted(self.node2id.values())

    def a_random_walk(self, node, length):
        if node not in self.graph:
            return []
        if not hasattr(self, 'node_choices'):
            self.create_node_choices()
        walk = [node]
        length -= 1
        while length > 0:
            # TODO fix mistake: next_node may be not in graph, callling graph[node] make error
            # Original
            '''
            if len(self.graph[node]) == 0:
                return walk
            '''
            # Modified
            #*******************
            if node not in self.graph:
                return walk
            if len(self.graph[node]) == 0:
                return walk
            #*******************
            next_node, edge_class_id =random.choice(self.node_choices[node])
            walk.append(edge_class_id)
            walk.append(next_node)
            node = next_node
            length -= 1
        return walk

    def create_node_choices(self):
        node_choices = {} #{from_id: [(to_id, edge_class_id)]}
        for from_id in self.graph:
            node_choices[from_id] = []
            for to_id in self.graph[from_id]:
                for edge_id, w in self.graph[from_id][to_id].items():
                    node_choices[from_id] += [(to_id, edge_id)] * int(w*10)
        self.node_choices = node_choices

    def random_walks(self, count, length, seed=None):
        '''
            Generate random walks starting from each node

            input:
                count: the # of random walks starting from each node
                length: the maximam length of a random walk

            output:
                [random_walk]
                    random_walk: [<node_id>,<edge_class_id>,<node_id>, ...]
        '''

        random.seed(seed)
        np.random.seed(seed)

        if not hasattr(self, 'node_choices'):
            self.create_node_choices()

        for c in range(count):
#           print c
            n = 0
            for node in self.graph:
                n += 1
                if n % 10000 == 0:
                    print(n)
                walk = self.a_random_walk(node, length)
                if len(walk) != 1:
                    yield walk

    #FIXME
    def random_select_edges_by_classes(self, from_class, to_class, edge_class, count):
        edge_class_id = self.edge_class2id[edge_class]
        positives = []
        for from_id in self.class_nodes[from_class]:
            for to_id in self.graph[from_id].get(edge_class_id, {}).keys():
                positives.append((from_id, to_id))
        random.shuffle(positives)
        selected_pos = positives[:count]

        positives = set(positives)
        from_nodes = list(self.class_nodes[from_class])
        to_nodes = list(self.class_nodes[to_class])
        from_node_count = len(from_nodes)
        to_node_count = len(to_nodes)
        selected_negs = set()
        while len(selected_negs) < count:
            rand_from = from_nodes[random.randint(0, from_node_count-1)]
            rand_to = to_nodes[random.randint(0, to_node_count-1)]
            if rand_to in self.graph[rand_from].get(edge_class_id, {}):
                continue
            selected_negs.add((rand_from, rand_to))
        return selected_pos, list(selected_negs)

    def in_k_hop_neighborhood(self, id1, id2, k):
        if not hasattr(self, 'k_hop_neighbors'):
            self.k_hop_neighbors = {}
        if k not in self.k_hop_neighbors:
            self.k_hop_neighbors[k] = {}
        if id1 not in self.k_hop_neighbors[k]:
            self.k_hop_neighbors[k][id1] = self._get_k_hop_neighborhood(id1, k)
        return id2 in self.k_hop_neighbors[k][id1]

    def _get_k_hop_neighborhood(self, id_, k):
        if not hasattr(self, 'k_hop_neighbors'):
            self.k_hop_neighbors = {}
        if k not in self.k_hop_neighbors:
            self.k_hop_neighbors[k] = {}
        if id_ in self.k_hop_neighbors[k]:
            return self.k_hop_neighbors[k][id_]

        neighbors = set()
        visited = set()
        to_visit = set([id_])
        next_to_visit = set()
        i = 0
        while i < k:
            while len(to_visit) != 0:
                visit_id = to_visit.pop()
                visited.add(visit_id)
                for to_id in self.graph[visit_id]:
                    if to_id == id_:
                        continue
                    if to_id in neighbors:
                        continue
                    neighbors.add(to_id)

                    if to_id in to_visit:
                        continue
                    if to_id not in visited:
                        next_to_visit.add(to_id)
            to_visit = next_to_visit.copy()
            i += 1
        self.k_hop_neighbors[k][id_] = neighbors
        return neighbors

    def get_candidates(self, node_id, k, to_node_class):
        khops = self._get_k_hop_neighborhood(node_id, k)
        onehops = self._get_k_hop_neighborhood(node_id, 1)
        khops = khops.difference(onehops)
        khops = khops.intersection(self.class_nodes[to_node_class])
        return khops

    def get_shortest_distance(self, node_id1, node_id2, max_=None):
        visited = set()
        to_visit = set([node_id1])
        nexts = set()
        distance = 1
        while True:
            for id_ in to_visit:
                for to_id in self.graph[id_]:
                    if to_id == node_id2:
                        return distance

                    if to_id in visited or to_id in to_visit:
                        continue

                    nexts.add(to_id)
                    visited.add(id_)

            if max_ is not None and distance == max_:
                return None

            if len(to_visit) == 0:
                return None

            distance += 1
            to_visit = nexts
            nexts = set()

        return None

    def random_remove_edges(self, edge_class, ratio=0.5, seed=None):
        random.seed(seed)
        edge_class_id = self.edge_class2id[edge_class]
        inv_edge_class_mapping = self.get_edge_class_inverse_mappling()
        try:
            inv_edge_class_id = int(inv_edge_class_mapping[str(edge_class_id)])
        except:
            inv_edge_class_id = edge_class_id

        edges = []
        for from_id, tos in self.graph.items():
            for to_id, to_edges in tos.items():
                if edge_class_id not in to_edges:
                    continue
                edges.append((from_id, to_id))
        random.shuffle(edges)
        for from_id, to_id in edges[:int(len(edges)*ratio)]:
            try:
                self.graph[from_id][to_id].pop(edge_class_id)
                if len(self.graph[from_id][to_id]) == 0:
                    self.graph[from_id].pop(to_id)
                self.graph[to_id][from_id].pop(inv_edge_class_id)
                if len(self.graph[to_id][from_id]) == 0:
                    self.graph[to_id].pop(from_id)
            except:
                continue
        return edges[:int(len(edges)*ratio)]

    def random_select_neg_edges(self, edge_class, count, seed=None):
        random.seed(seed)
        edge_class_id = self.edge_class2id[edge_class]
        from_class, to_class = self.edge_class_id_available_node_class[edge_class_id]
        from_ids = list(self.class_nodes[from_class])
        from_id_size = len(from_ids)
        to_ids = list(self.class_nodes[to_class])
        to_id_size = len(to_ids)
        selected = set()
        while len(selected) < count:
            rand_from_id = from_ids[random.randint(0, from_id_size-1)]
            rand_to_id = to_ids[random.randint(0, to_id_size-1)]
            if (rand_from_id, rand_to_id) in selected:
                continue
            if (rand_to_id in self.graph[rand_from_id]
                and edge_class_id in self.graph[rand_from_id][rand_to_id]):
                continue
            selected.add((rand_from_id, rand_to_id))
            if len(selected) % 10000 == 0:
                print(len(selected))
        return selected

    def to_edge_class_id_string(self, edge_classes):
        return ','.join(map(str, [self.edge_class2id[ec] for ec in edge_classes]))

    def generate_test_set(self, path, count, seed=None):
        random.seed(seed)

        from_class, _ = self.edge_class_id_available_node_class[path[0]]
        from_ids = list(self.class_nodes[from_class])
        _, to_class = self.edge_class_id_available_node_class[path[-1]]
        to_ids = list(self.class_nodes[to_class])

        test_set = []
        while len(test_set) < count:
            from_id = from_ids[random.randint(0, len(from_ids)-1)]
            to_id = to_ids[random.randint(0, len(to_ids)-1)]
            if from_id == to_id:
                continue

            label = 1 if self._check_path(from_id, to_id, path) else 0
            test_set.append((from_id, to_id, label))
        return test_set

    def _check_path(self, from_id, to_id, path):
        from_ids = set([from_id])
        for edge in path:
            next_ids = set()
            for from_id in from_ids:
                for next_id in self.graph[from_id]:
                    if edge in self.graph[from_id][next_id]:
                        next_ids.add(next_id)
            from_ids = next_ids
        return to_id in from_ids
