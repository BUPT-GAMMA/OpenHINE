import random
import numpy as np

__all__ = [
    'random_walk_based_mp',
    'random_walk_based_seq',
    'random_walk_spacey_mp',
    'MetaGraphGenerator']


def random_walk_based_mp(g, metapath, num_walks, walk_length, outfilename):
    # MP based RW is only applicable for metapaths with same source and target nodes.
    try:
        if metapath[0] != metapath[-1]:
            raise ValueError
    except ValueError:
        print("It is only applicable for metapaths with same source and target nodes.")

    print("Metapath walking!")
    relation_list = []
    for i in range(len(metapath) - 1):
        relation_list.append(g.relation_dict[metapath[i] + '-' + metapath[i + 1]])
    start_list = g.node[metapath[0]]

    with open(outfilename, 'w') as outfile:
        for start_node in start_list:
            for _ in range(num_walks):
                outline = metapath[0] + start_node
                current_node = start_node
                l, r = 1, 1
                while l < walk_length:
                    current_node = random.choice(relation_list[r - 1][current_node])
                    outline += " " + metapath[r] + current_node
                    r = r + 1 if r + 1 < len(metapath) else 1
                    l += 1
                outfile.write(outline + "\n")
    print("Finish walking!")

def random_walk_based_mp_personalize(g, metapath, num_walks, walk_length, outfilename):
    # MP based RW is only applicable for metapaths with same source and target nodes.
    try:
        if metapath[0] != metapath[-1]:
            raise ValueError
    except ValueError:
        print("It is only applicable for metapaths with same source and target nodes.")

    print("Metapath walking!")
    relation_list = []
    for i in range(len(metapath) - 1):
        relation_list.append(g.relation_dict[metapath[i] + '-' + metapath[i + 1]])
    start_list = g.node[metapath[0]]

    with open(outfilename, 'w') as outfile:
        for start_node in sorted(start_list):
            for _ in range(num_walks):
                outline = metapath[0] + start_node
                current_node = start_node
                l, r = 1, 1
                while l < walk_length:
                    if relation_list[r - 1][current_node] == None:
                        continue
                    else:
                        current_node = random.choice(relation_list[r - 1][current_node])
                        outline += " " + metapath[r] + current_node
                        r = r + 1 if r + 1 < len(metapath) else 1
                        l += 1
                outfile.write(outline + "\n")
    print("Finish walking!")

def random_walk_based_seq(g, seq, num_walks=1, outflilename=None):
    print("Start walking!")
    start_list = g.node[seq[0]]
    with open(outflilename, 'w') as outfile:
        for start_node in sorted(start_list):
            for _ in range(num_walks):
                outline = seq[0] + start_node
                current_node = start_node
                r = 1
                while r < len(seq):
                    relation = seq[r - 1] + '-' + seq[r]
                    try:
                        current_node = random.choice(g.relation_dict[relation][current_node])
                    except KeyError:
                        print("metapath illegal!")

                    outline += " " + seq[r] + current_node
                    r += 1
                outfile.write(outline + "\n")
    print("Finish walking!")


def random_walk_spacey_mp(g, metapath, mg_type, numwalks, walklength, outfilename,
                          beta):
    start_list = g.node[mg_type[0]]
    history = {mg_type[i]: 1 for i in range(len(mg_type))}
    adj_type_list = {}
    for i in range(len(mg_type)):
        adj_type_list[mg_type[i]] = []
    for i in range(len(metapath) - 1):
        if (metapath[i + 1] in adj_type_list[metapath[i]]):
            continue
        else:
            adj_type_list[metapath[i]] += metapath[i + 1]
    print("Spacey Metapath walking!")
    with open(outfilename, 'w') as outfile:
        for i in sorted(start_list):
            for j in range(0, numwalks):
                cur_node = i
                cur_type = mg_type[0]
                outline = cur_type + cur_node
                for k in range(walklength):
                    temp = cur_type
                    if random.random() < beta:
                        next_type_list = adj_type_list[cur_type]
                        if len(next_type_list) == 0:
                            break
                        elif len(next_type_list) == 1:
                            cur_type = next_type_list[0]
                            cur_node = random.choice(g.relation_dict[temp + '-' + cur_type][cur_node])
                        else:
                            occupancy = [history[i] for i in next_type_list]
                            weight_sum = np.sum(occupancy)
                            occupancy /= weight_sum
                            # 选择当前节点类型算法有问题
                            cur_type = np.random.choice(next_type_list, size=1, replace=True, p=occupancy)[0]
                            cur_node = random.choice(g.relation_dict[temp + '-' + cur_type][cur_node])
                        history[cur_type] += 1
                    else:
                        next_type_list = adj_type_list[cur_type]
                        if len(next_type_list) == 0:
                            break
                        else:
                            cur_type = random.choice(next_type_list)
                            cur_node = random.choice(g.relation_dict[temp + '-' + cur_type][cur_node])
                        history[cur_type] += 1
                    outline += " " + str(cur_type + cur_node)
                outfile.write(outline + "\n")
    print("Finish walking!")

def generate_spacey_metaschema_metagraph_random_walk(self, relations, mg_type, outfilename, numwalks, walklength,
                                                     datasource, beta):
    start_list = datasource.node[mg_type[0]]
    history = {mg_type[i]: 1 for i in range(len(mg_type))}
    with open(outfilename, 'w') as outfile:
        for i in start_list:
            for j in range(0, numwalks):
                cur_node = i
                cur_type = mg_type[0]
                outline = cur_type + cur_node
                for k in range(walklength):
                    temp = cur_type
                    if random.random() < beta:
                        next_type_list = relations[cur_type]
                        if len(next_type_list) == 0:
                            break
                        elif len(next_type_list) == 1:
                            cur_type = next_type_list[0]
                            cur_node = random.choice(datasource.relation_dict[temp + '-' + cur_type][cur_node])
                        else:
                            occupancy = [history[i] for i in next_type_list]
                            weight_sum = np.sum(occupancy)
                            occupancy /= weight_sum
                            # 选择当前节点类型算法有问题
                            cur_type = np.random.choice(next_type_list, size=1, replace=True, p=occupancy)[0]
                            cur_node = random.choice(datasource.relation_dict[temp + '-' + cur_type][cur_node])
                        history[cur_type] += 1
                    else:
                        next_type_list = relations[cur_type]
                        if len(next_type_list) == 0:
                            break
                        else:
                            cur_type = random.choice(next_type_list)
                            cur_node = random.choice(datasource.relation_dict[temp + '-' + cur_type][cur_node])
                        history[cur_type] += 1
                    outline += " " + str(cur_type + cur_node)
                outfile.write(outline + "\n")


class MetaGraphGenerator:
    def __init__(self):
        pass

    # walk length=80  walks per node=40
    def generate_random_four(self, outfilename, numwalks, walklength,
                             nodelist, relation_dict):
        mg_type = "apct"
        start_list = nodelist[mg_type[0]]
        relation1 = relation_dict["a-p"]
        relation2 = relation_dict["p-a"]
        relation3 = relation_dict["p-c"]
        relation4 = relation_dict["c-p"]

        relation5 = relation_dict["p-t"]
        relation6 = relation_dict["t-p"]

        with open(outfilename, 'w') as outfile:
            for i in start_list:
                for j in range(0, numwalks):
                    k = 0
                    current_node = mg_type[0] + i
                    outline = current_node
                    while k < walklength:
                        if (current_node[0] == mg_type[0]):
                            current_node = mg_type[1] + random.choice(relation1[current_node[1:]])
                            outline += " " + current_node
                        elif current_node[0] == mg_type[1]:
                            l1 = len(relation2[current_node[1:]])
                            l2 = len(relation3[current_node[1:]])
                            l3 = len(relation5[current_node[1:]])
                            random_int = random.randint(1, l1 + l2 + l3)
                            if random_int <= l1:
                                current_node = mg_type[0] + random.choice(relation2[current_node[1:]])
                            elif random_int <= l1 + l2:
                                current_node = mg_type[2] + random.choice(relation3[current_node[1:]])
                            else:
                                current_node = mg_type[3] + random.choice(relation5[current_node[1:]])
                            outline += " " + current_node
                        elif current_node[0] == mg_type[2]:
                            current_node = mg_type[1] + random.choice(relation4[current_node[1:]])
                            outline += " " + current_node
                        elif current_node[0] == mg_type[3]:
                            current_node = mg_type[1] + random.choice(relation6[current_node[1:]])
                            outline += " " + current_node
                        else:
                            print("error!")
                        k += 1
                    outfile.write(outline + "\n")

    def generate_random_three(self, outfilename, numwalks, walklength,
                              nodelist, relation_dict):
        mg_type = "aps"
        start_list = nodelist[mg_type[0]]
        relation1 = relation_dict["a-p"]
        relation2 = relation_dict["p-a"]
        relation3 = relation_dict["p-s"]
        relation4 = relation_dict["s-p"]

        with open(outfilename, 'w') as outfile:
            for i in start_list:
                for j in range(0, numwalks):
                    k = 0
                    current_node = mg_type[0] + i
                    outline = current_node
                    while k < walklength:
                        if (current_node[0] == mg_type[0]):
                            current_node = mg_type[1] + random.choice(relation1[current_node[1:]])
                            outline += " " + current_node
                        elif current_node[0] == mg_type[1]:
                            l1 = len(relation2[current_node[1:]])
                            l2 = len(relation3[current_node[1:]])
                            random_int = random.randint(1, l1 + l2)
                            if random_int <= l1:
                                current_node = mg_type[0] + random.choice(relation2[current_node[1:]])
                            else:
                                current_node = mg_type[2] + random.choice(relation3[current_node[1:]])
                            outline += " " + current_node
                        elif current_node[0] == mg_type[2]:
                            current_node = mg_type[1] + random.choice(relation4[current_node[1:]])
                            outline += " " + current_node
                        else:
                            print("error!")
                        k += 1
                    outfile.write(outline + "\n")
