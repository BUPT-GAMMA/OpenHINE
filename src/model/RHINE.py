from __future__ import division
import warnings
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import ctypes
import json
from torch.autograd import Variable
import os

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, normalized_mutual_info_score

warnings.filterwarnings('ignore')


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

    def get_postive_IRs(self):
        """
        sample positive IRs triples
        :return:
        """
        self.postive_h_IRs = Variable(torch.from_numpy(
            self.config.batch_h_IRs[0:int(self.config.batch_size_IRs)])).cuda()
        self.postive_t_IRs = Variable(torch.from_numpy(
            self.config.batch_t_IRs[0:int(self.config.batch_size_IRs)])).cuda()
        self.postive_r_IRs = Variable(torch.from_numpy(
            self.config.batch_r_IRs[0:int(self.config.batch_size_IRs)])).cuda()
        self.postive_w_IRs = Variable(torch.from_numpy(
            self.config.batch_w_IRs[0:int(self.config.batch_size_IRs)])).cuda()
        return self.postive_h_IRs, self.postive_t_IRs, self.postive_r_IRs, self.postive_w_IRs

    def get_negtive_IRs(self):
        """
        sample negative IRs triples
        :return:
        """
        self.negtive_h_IRs = Variable(
            torch.from_numpy(self.config.batch_h_IRs[int(self.config.batch_size_IRs):int(self.config.batch_seq_size_IRs)])).cuda()
        self.negtive_t_IRs = Variable(
            torch.from_numpy(self.config.batch_t_IRs[int(self.config.batch_size_IRs):int(self.config.batch_seq_size_IRs)])).cuda()
        self.negtive_r_IRs = Variable(
            torch.from_numpy(self.config.batch_r_IRs[int(self.config.batch_size_IRs):int(self.config.batch_seq_size_IRs)])).cuda()
        self.negtive_w_IRs = Variable(
            torch.from_numpy(self.config.batch_w_IRs[int(self.config.batch_size_IRs):int(self.config.batch_seq_size_IRs)])).cuda()

        return self.negtive_h_IRs, self.negtive_t_IRs, self.negtive_r_IRs, self.negtive_w_IRs

    def get_postive_ARs(self):
        self.postive_h_ARs = Variable(torch.from_numpy(
            self.config.batch_h_ARs[0:int(self.config.batch_size_ARs)])).cuda()
        self.postive_t_ARs = Variable(torch.from_numpy(
            self.config.batch_t_ARs[0:int(self.config.batch_size_ARs)])).cuda()
        self.postive_r_ARs = Variable(torch.from_numpy(
            self.config.batch_r_ARs[0:int(self.config.batch_size_ARs)])).cuda()
        self.postive_w_ARs = Variable(torch.from_numpy(
            self.config.batch_w_ARs[0:int(self.config.batch_size_ARs)])).cuda()
        return self.postive_h_ARs, self.postive_t_ARs, self.postive_r_ARs, self.postive_w_ARs

    def get_negtive_ARs(self):
        self.negtive_h_ARs = Variable(
            torch.from_numpy(self.config.batch_h_ARs[int(self.config.batch_size_ARs):int(self.config.batch_seq_size_ARs)])).cuda()
        self.negtive_t_ARs = Variable(
            torch.from_numpy(self.config.batch_t_ARs[int(self.config.batch_size_ARs):int(self.config.batch_seq_size_ARs)])).cuda()
        self.negtive_r_ARs = Variable(
            torch.from_numpy(self.config.batch_r_ARs[int(self.config.batch_size_ARs):int(self.config.batch_seq_size_ARs)])).cuda()
        self.negtive_w_ARs = Variable(
            torch.from_numpy(self.config.batch_w_ARs[int(self.config.batch_size_ARs):int(self.config.batch_seq_size_ARs)])).cuda()

        return self.negtive_h_ARs, self.negtive_t_ARs, self.negtive_r_ARs, self.negtive_w_ARs

    def predict(self):
        pass

    def forward(self):
        pass

    def loss_func(self):
        pass


class RHINE(Model):

    def __init__(self, config):
        super(RHINE, self).__init__(config)
        self.ent_embeddings = nn.Embedding(
            config.total_nodes, config.hidden_size)
        self.rel_embeddings = nn.Embedding(
            config.total_IRs, config.hidden_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform(self.rel_embeddings.weight.data)

    def translation_dis(self, h, t, r):
        return torch.abs(h + r - t)

    def euclidea_dis(self, e, v):
        return torch.pow(e - v, 2)

    def loss_func(self, p_score, n_score):
        criterion = nn.MarginRankingLoss(self.config.margin, False).cuda()
        y = Variable(torch.Tensor([-1])).cuda()
        loss = criterion(p_score, n_score, y)
        return loss

    def forward(self, mode):
        loss = 0
        if mode == 'Trans':
            pos_h, pos_t, pos_r, pos_rel_w = self.get_postive_IRs()
            neg_h, neg_t, neg_r, neg_rel_w = self.get_negtive_IRs()
            neg_rate = len(neg_h) / len(pos_h)
            neg_step = len(pos_h)

            p_h = self.ent_embeddings(pos_h)
            p_t = self.ent_embeddings(pos_t)
            p_r = self.rel_embeddings(pos_r)
            n_h = self.ent_embeddings(neg_h)
            n_t = self.ent_embeddings(neg_t)
            n_r = self.rel_embeddings(neg_r)
            _p_score = self.translation_dis(p_h, p_t, p_r)
            _n_score = self.translation_dis(n_h, n_t, n_r)
            p_score = torch.sum(_p_score, 1)
            n_score = torch.sum(_n_score, 1)
            pos_rel_w = pos_rel_w.float()
            neg_rel_w = neg_rel_w.float()
            trans_loss = 0
            for i in range(int(neg_rate)):
                trans_loss += self.loss_func(pos_rel_w * p_score,
                                             neg_rel_w[i * neg_step:(i + 1) * neg_step] * n_score[i * neg_step:(i + 1) * neg_step])
            loss = trans_loss
        elif mode == 'Euc':
            pos_e, pos_v, pos_a, pos_attr_w = self.get_postive_ARs()
            neg_e, neg_v, neg_a, neg_attr_w = self.get_negtive_ARs()
            neg_rate = len(neg_e) / len(pos_e)
            neg_step = len(pos_e)

            p_e = self.ent_embeddings(pos_e)
            p_v = self.ent_embeddings(pos_v)
            n_e = self.ent_embeddings(neg_e)
            n_v = self.ent_embeddings(neg_v)
            _p_score = self.euclidea_dis(p_e, p_v)
            _n_score = self.euclidea_dis(n_e, n_v)
            p_score = torch.sum(_p_score, 1)
            n_score = torch.sum(_n_score, 1)
            pos_attr_w = pos_attr_w.float()
            neg_attr_w = neg_attr_w.float()
            cl_loss = 0
            for i in range(int(neg_rate)):
                # cl_loss = self.cl_loss_func(norm_pos_attr_w*p_score, norm_neg_attr_w*n_score)
                cl_loss += self.loss_func(pos_attr_w * p_score,
                                          neg_attr_w[i * neg_step:(i + 1) * neg_step] * n_score[i * neg_step:(i + 1) * neg_step])
            loss = cl_loss
        return loss


def TrainRHINE(config, node2dict):
    con = RHINEConfig()
    con.set_node2dict(node2dict)
    con.set_in_path(config.temp_file)
    con.set_work_threads(config.num_workers)
    con.set_train_times(config.epochs)
    con.set_IRs_nbatches(config.IRs_nbatches)
    con.set_ARs_nbatches(config.ARs_nbatches)
    con.set_alpha(config.alpha)
    con.set_margin(config.margin)
    con.set_dimension(config.dim)
    con.set_ent_neg_rate(config.ent_neg_rate)
    con.set_rel_neg_rate(config.rel_neg_rate)
    con.set_opt_method(config.opt_method)
    con.set_optimizer(config.optimizer)
    con.set_evaluation(config.evaluation_flag)
    con.set_exportName(config.exportName)
    con.set_importName(config.importName)
    con.set_lr_decay(config.lr_decay)
    con.set_log_on(config.log_on)
    con.set_weight_decay(config.weight_decay)
    con.set_export_steps(config.export_steps)
    con.set_export_files(str(config.output_modelfold) +
                         "/RHINE" + "/model.vec." + str(config.mode) + ".tf")
    con.set_out_files(config.out_emd_file + "node.txt")

    # print(con)
    con.init()
    # print(con)
    con.set_model(RHINE)
    # print(con)
    con.run()

    # print('evaluation...')
    # exp = Evaluation()
    # emb_dict = exp.load_emb(config.out_emd_file +
    #                         "/embedding.vec." +
    #                         str(config.mode) +
    #                         ".json")
    # print(emb_dict)
    # exp.evaluation(emb_dict)


class RHINEConfig(object):

    def __init__(self):
        self.lib_IRs = ctypes.cdll.LoadLibrary("./src/release/Sample_IRs.so")
        self.lib_IRs.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        self.lib_IRs.getHeadBatch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib_IRs.getTailBatch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

        self.lib_ARs = ctypes.cdll.LoadLibrary("./src/release/Sample_ARs.so")
        self.lib_ARs.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        self.lib_ARs.getHeadBatch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib_ARs.getTailBatch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

    def init(self):
        """
        init. parameters
        :return:
        """
        self.trainModel = None
        # print(self.in_path)
        if self.in_path is not None:
            print(self.in_path)
            # sample IRs
            b = bytes(self.in_path, encoding='utf-8')
            # print(len(self.in_path))
            self.lib_IRs.setInPath(
                ctypes.create_string_buffer(
                    b, len(
                        self.in_path) * 2))
            self.lib_IRs.setWorkThreads(self.workThreads)
            self.lib_IRs.randReset()
            # print("YES")
            self.lib_IRs.importTrainFiles()

            self.total_IRs = self.lib_IRs.getRelationTotal()
            self.total_nodes = self.lib_IRs.getEntityTotal()
            self.train_total_IRs_triple = self.lib_IRs.getTrainTotal()
            self.batch_size_IRs = self.lib_IRs.getTrainTotal() / self.IRs_nbatches
            print('# IRs triples: {}'.format(self.train_total_IRs_triple))
            print('IRs triple batch size: {}'.format(self.batch_size_IRs))
            self.batch_seq_size_IRs = self.batch_size_IRs * \
                (1 + self.negative_ent + self.negative_rel)
            self.batch_h_IRs = np.zeros(int(
                self.batch_size_IRs * (1 + self.negative_ent + self.negative_rel)), dtype=np.int64)
            self.batch_t_IRs = np.zeros(int(
                self.batch_size_IRs * (1 + self.negative_ent + self.negative_rel)), dtype=np.int64)
            self.batch_r_IRs = np.zeros(int(
                self.batch_size_IRs * (1 + self.negative_ent + self.negative_rel)), dtype=np.int64)
            self.batch_w_IRs = np.ones(int(
                self.batch_size_IRs * (1 + self.negative_ent + self.negative_rel)), dtype=np.int64)

            self.batch_h_addr_IRs = self.batch_h_IRs.__array_interface__[
                'data'][0]
            self.batch_t_addr_IRs = self.batch_t_IRs.__array_interface__[
                'data'][0]
            self.batch_r_addr_IRs = self.batch_r_IRs.__array_interface__[
                'data'][0]
            self.batch_w_addr_IRs = self.batch_w_IRs.__array_interface__[
                'data'][0]

            # sample ARs
            self.lib_ARs.setInPath(
                ctypes.create_string_buffer(
                    b, len(
                        self.in_path) * 2))
            self.lib_ARs.setWorkThreads(self.workThreads)
            self.lib_ARs.randReset()
            self.lib_ARs.importTrainFiles()
            self.train_total_ARs_triple = self.lib_ARs.getTrainTotal()
            self.batch_size_ARs = self.lib_ARs.getTrainTotal() / self.ARs_nbatches
            print('# ARs triples: {}'.format(self.train_total_ARs_triple))
            print('ARs triple batch size: {}'.format(self.batch_size_ARs))
            self.batch_seq_size_ARs = self.batch_size_ARs * \
                (1 + self.negative_ent + self.negative_rel)
            self.batch_h_ARs = np.zeros(int(
                self.batch_size_ARs * (1 + self.negative_ent + self.negative_rel)), dtype=np.int64)
            self.batch_t_ARs = np.zeros(int(
                self.batch_size_ARs * (1 + self.negative_ent + self.negative_rel)), dtype=np.int64)
            self.batch_r_ARs = np.zeros(int(
                self.batch_size_ARs * (1 + self.negative_ent + self.negative_rel)), dtype=np.int64)
            self.batch_w_ARs = np.ones(int(
                self.batch_size_ARs * (1 + self.negative_ent + self.negative_rel)), dtype=np.int64)

            self.batch_h_addr_ARs = self.batch_h_ARs.__array_interface__[
                'data'][0]
            self.batch_t_addr_ARs = self.batch_t_ARs.__array_interface__[
                'data'][0]
            self.batch_r_addr_ARs = self.batch_r_ARs.__array_interface__[
                'data'][0]
            self.batch_w_addr_ARs = self.batch_w_ARs.__array_interface__[
                'data'][0]

    def set_node2dict(self, node2dict):
        self.node2dict = node2dict

    def set_opt_method(self, method):
        self.opt_method = method

    def set_log_on(self, flag):
        self.log_on = flag

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_in_path(self, path):
        self.in_path = path

    def set_out_files(self, path):
        self.out_path = path

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_dimension(self, dim):
        self.hidden_size = dim
        self.ent_size = dim
        self.rel_size = dim

    def set_train_times(self, times):
        self.train_times = times

    def set_IRs_nbatches(self, nbatches):
        self.IRs_nbatches = nbatches

    def set_ARs_nbatches(self, nbatches):
        self.ARs_nbatches = nbatches

    def set_margin(self, margin):
        self.margin = margin

    def set_work_threads(self, threads):
        self.workThreads = threads

    def set_ent_neg_rate(self, rate):
        self.negative_ent = rate

    def set_rel_neg_rate(self, rate):
        self.negative_rel = rate

    def set_import_files(self, path):
        self.importName = path

    def set_export_files(self, path):
        self.exportName = path

    def set_export_steps(self, steps):
        self.export_steps = steps

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_evaluation(self, evaluation_flag):
        self.evaluation_flag = evaluation_flag

    def set_exportName(self, exportName):
        self.exportName = exportName

    def set_importName(self, importName):
        self.importName = importName

    def sampling_IRs(self):
        self.lib_IRs.sampling(self.batch_h_addr_IRs, self.batch_t_addr_IRs, self.batch_r_addr_IRs, self.batch_w_addr_IRs,
                              int(self.batch_size_IRs), self.negative_ent, self.negative_rel)

    def sampling_ARs(self):
        self.lib_ARs.sampling(self.batch_h_addr_ARs, self.batch_t_addr_ARs, self.batch_r_addr_ARs, self.batch_w_addr_ARs,
                              int(self.batch_size_ARs), self.negative_ent, self.negative_rel)

    def save_pytorch(self):
        torch.save(self.trainModel.state_dict(), self.exportName)

    def restore_pytorch(self):
        self.trainModel.load_state_dict(torch.load(self.importName))

    def get_parameter_lists(self):
        return self.trainModel.cpu().state_dict()

    def get_parameters(self, mode="numpy"):
        res = {}
        lists = self.get_parameter_lists()
        for var_name in lists:
            if mode == "numpy":
                res[var_name] = lists[var_name].numpy()
            if mode == "list":
                res[var_name] = lists[var_name].numpy().tolist()
            else:
                res[var_name] = lists[var_name]
        return res

    def save_parameters(self, path=None):
        if path is None:
            path = self.out_path
        res = self.get_parameter_lists()['ent_embeddings.weight'].numpy().tolist()
        embedding_dict = {}
        for node in self.node2dict:
            embedding_dict[node] = res[self.node2dict[node]]
        n_node = len(embedding_dict)
        dim = len(res[0])
        with open(path, 'w') as f:
            embedding_str = str(n_node) + '\t' + str(dim) + '\n'
            for emd in embedding_dict:
                embedding_str += str(emd) + ' ' + ' '.join([str(x) for x in embedding_dict[emd]]) + '\n'
            f.write(embedding_str)
        # f = open(path, "w")
        # f.write(json.dumps(self.get_parameters("list")))
        # f.close()

    def set_parameters_by_name(self, var_name, tensor):
        self.trainModel.state_dict().get(var_name).copy_(
            torch.from_numpy(np.array(tensor)))

    def set_parameters(self, lists):
        for i in lists:
            self.set_parameters_by_name(i, lists[i])

    def set_model(self, model):
        self.model = model
        self.trainModel = self.model(config=self)
        self.trainModel.cuda()
        if self.optimizer is not None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(self.trainModel.parameters(), lr=self.alpha,
                                           lr_decay=self.lr_decay, weight_decay=self.weight_decay)
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.trainModel.parameters(), lr=self.alpha)
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.trainModel.parameters(), lr=self.alpha)
        else:
            self.optimizer = optim.SGD(
                self.trainModel.parameters(), lr=self.alpha)

    def run(self):
        if self.importName is not None:
            self.restore_pytorch()
        for epoch in range(self.train_times):
            res = 0.0
            for batch in range(self.IRs_nbatches):
                self.sampling_IRs()
                self.optimizer.zero_grad()
                loss = self.trainModel('Trans')
                res += loss.item()
                loss.backward()
                self.optimizer.step()
            for batch in range(self.ARs_nbatches):
                self.sampling_ARs()
                self.optimizer.zero_grad()
                loss = self.trainModel('Euc')
                res += loss.item()
                loss.backward()
                self.optimizer.step()

            if self.exportName is not None and (
                    self.export_steps != 0 and epoch % self.export_steps == 0):
                self.save_pytorch()
            if self.log_on == 1:
                print('Epoch: {}, loss: {}'.format(epoch, res))
            # if self.evaluation_flag and epoch != 0 and epoch % 100 == 0:
            #     emb_json = self.get_parameters("list")
            #     evaluation(emb_json)
            #     self.trainModel.cuda()

        if self.out_path is not None:
            self.save_parameters(self.out_path)

#
# class Evaluation:
#     def __init__(self):
#         self.entity_name_emb_dict = {}
#         np.random.seed(1)
#
#     def load_emb(self, emb_name):
#         """
#         load embeddings
#         :param emb_name:
#         :return:
#         """
#         with open(emb_name, 'r') as emb_file:
#             emb_dict = json.load(emb_file)
#         return emb_dict
#
#     def evaluation(self, emb_dict):
#         entity_emb = emb_dict['ent_embeddings.weight']
#         with open('../data/dblp/node2id.txt', 'r') as e2i_file:
#             lines = e2i_file.readlines()
#
#         paper_id_name_dict = {}
#         for i in range(1, len(lines)):
#             tokens = lines[i].strip().split('\t')
#             if lines[i][0] == 'p':
#                 paper_id_name_dict[tokens[1]] = tokens[0]
#
#         for p_id, p_name in paper_id_name_dict.items():
#             p_emb = map(lambda x: float(x), entity_emb[int(p_id)])
#             self.entity_name_emb_dict[p_name] = p_emb
#
#         x_paper = []
#         y_paper = []
#         with open('../data/dblp/paper_label.txt', 'r') as paper_name_label_file:
#             paper_name_label_lines = paper_name_label_file.readlines()
#         for line in paper_name_label_lines:
#             tokens = line.strip().split('\t')
#             x_paper.append(self.entity_name_emb_dict['p' + tokens[0]])
#             y_paper.append(int(tokens[1]))
#         self.kmeans_nmi(x_paper, y_paper, k=4)
#         self.classification(x_paper, y_paper)
#
#     def kmeans_nmi(self, x, y, k):
#         km = KMeans(n_clusters=k)
#         km.fit(x, y)
#         y_pre = km.predict(x)
#
#         nmi = normalized_mutual_info_score(y, y_pre)
#         print('NMI: {}'.format(nmi))
#
#     def classification(self, x, y):
#         x_train, x_valid, y_train, y_valid = train_test_split(
#             x, y, test_size=0.2, random_state=9)
#
#         lr = LogisticRegression()
#         lr.fit(x_train, y_train)
#
#         y_valid_pred = lr.predict(x_valid)
#         micro_f1 = f1_score(y_valid, y_valid_pred, average='micro')
#         macro_f1 = f1_score(y_valid, y_valid_pred, average='macro')
#         print('Macro-F1: {}'.format(macro_f1))
#         print('Micro-F1: {}'.format(micro_f1))