import numpy as np


def HERec_union_metapth(input_fold, mp_list, n_node, dim):
    print("union metapath!")
    mtx_list = []
    for mp in mp_list:
        file = input_fold + mp + ".txt"
        mtx_list.append(read_embeddings(file, n_node, dim))
    index = np.array(range(n_node)).reshape(-1, 1)
    embedding_mtx = mtx_list[0]
    embedding_mtx = np.hstack([index, embedding_mtx])
    if len(mtx_list) > 1:
        for n in range(1, len(mtx_list)):
            embedding_mtx = np.concatenate((embedding_mtx, mtx_list[n]), axis=1)
    embedding_list = embedding_mtx.tolist()
    embedding_str = [mp_list[0][0] + str(int(emb[0])) + ' ' + ' '.join([str(x) for x in emb[1:]]) + '\n' for emb in
                     embedding_list]
    dim = dim * len(mp_list)
    with open(input_fold + "union_mp.txt", "w") as f:
        lines = [str(n_node) + ' ' + str(dim) + '\n'] + embedding_str
        f.writelines(lines)

def write_emd_file(filename, embedding, dim):
    with open(filename, 'w') as f:
        embedding_str = str(len(embedding)) + '\t' + str(dim) + '\n'
        for id, emd in embedding.items():
            embedding_str += str(id) + ' ' + ' '.join([str(x) for x in emd]) + '\n'
        f.write(embedding_str)

def read_embeddings(filename, n_node, n_embed):
    embedding_matrix = np.random.rand(n_node, n_embed)
    i = -1
    with open(filename) as infile:
        for line in infile.readlines()[1:]:
            i += 1
            emd = line.strip().split()
            embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
    return embedding_matrix

def read_embeddings_with_type(filename, n_node, n_embed, node2id):
    embedding_matrix = np.random.rand(n_node, n_embed)
    i = -1
    with open(filename) as infile:
        for line in infile.readlines()[1:]:
            i += 1
            emd = line.strip().split()
            embedding_matrix[int(node2id[emd[0]]), :] = str_list_to_float(emd[1:])
    return embedding_matrix


def str_list_to_float(str_list):
    return [float(item) for item in str_list]
