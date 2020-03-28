import optparse
import os
import sys
import tempfile
import src.model.network as network
from src.model.mp2vec_s import MP2Vec
__author__ = 'sheep'


def HIN2vec(graph_fname, output_datafold, options):
    '''\
    %prog [options] <graph_fname> <node_vec_fname> <path_vec_fname>

    graph_fname: the graph file
        It can be a file contained edges per line (e.g., res/karate_club_edges.txt)
        or a pickled graph file.
    node_vec_fname: the output file for nodes' vectors
    path_vec_fname: the output file for meta-paths' vectors
    '''
    node_vec_fname = output_datafold + "node.txt"
    path_vec_fname = output_datafold + "metapath.txt"
    options.allow_circle = False
    options.correct_neg = False
    print('Load a HIN...')
    g = load_a_HIN(graph_fname)

    print('Generate random walks...')
    _, tmp_walk_fname = tempfile.mkstemp()
    # print(tmp_walk_fname)
    with open(tmp_walk_fname, 'w') as f:
        for walk in g.random_walks(options.num_walks, options.walk_length):
            f.write('%s\n' % ' '.join(map(str, walk)))

    _, tmp_node_vec_fname = tempfile.mkstemp()
    _, tmp_path_vec_fname = tempfile.mkstemp()

    model = MP2Vec(size=options.dim,
                   window=options.window_size,
                   neg=options.neg_num,
                   num_processes=options.num_workers,
                   #                  iterations=i,
                   alpha=options.alpha,
                   same_w=True,
                   normed=False,
                   is_no_circle_path=False,
                   )

    neighbors = None
    if options.correct_neg:
        for id_ in g.graph:
            g._get_k_hop_neighborhood(id_, options.window_size)
        neighbors = g.k_hop_neighbors[options.window_size]

    model.train(g,
                tmp_walk_fname,
                g.class_nodes,
                k_hop_neighbors=neighbors,
                )
    model.dump_to_file(tmp_node_vec_fname, type_='node')
    model.dump_to_file(tmp_path_vec_fname, type_='path')

    print('Dump vectors...')
    output_node2vec(g, tmp_node_vec_fname, node_vec_fname)
    output_path2vec(g, tmp_path_vec_fname, path_vec_fname)
    return 0


def output_node2vec(g, tmp_node_vec_fname, node_vec_fname):
    with open(tmp_node_vec_fname) as f:
        with open(node_vec_fname, 'w') as fo:
            id2node = dict([(v, k) for k, v in g.node2id.items()])
            first = True
            for line in f:
                if first:
                    first = False
                    fo.write(line)
                    continue

                id_, vectors = line.strip().split(' ', 1)
                line = '%s %s\n' % (id2node[int(id_)], vectors)
                fo.write(line)

#FIXME: to support more than 10 different meta-paths
def output_path2vec(g, tmp_path_vec_fname, path_vec_fname):
    with open(tmp_path_vec_fname) as f:
        with open(path_vec_fname, 'w') as fo:
            id2edge_class = dict([(v, k) for k, v
                                  in g.edge_class2id.items()])
            print(id2edge_class)
            first = True
            for line in f:
                if first:
                    first = False
                    fo.write(line)
                    continue

                ids, vectors = line.strip().split(' ', 1)
                ids = map(int, ids.split(','))
                edge = ','.join([id2edge_class[id_] for id_ in ids])
                line = '%s %s\n' % (edge, vectors)
                fo.write(line)

def load_a_HIN(fname):
    g = network.HIN()
    relation_dict = fname.relation_dict
    for relation in relation_dict:
        src_class = relation[0]
        dst_class = relation[2]
        edge_class = relation
        for src in relation_dict[relation]:
            for dst in relation_dict[relation][src]:
                g.add_edge(src_class+src, src_class, dst_class+dst, dst_class, edge_class)
    #g.print_statistics()
    return g