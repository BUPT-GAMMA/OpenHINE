import time
import numpy as np
import tensorflow as tf
import scipy.io as sio
from src.model.GAT import HeteGAT_multi
from src.utils.utils import write_emd_file
import warnings
warnings.filterwarnings(action='error')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# 禁用gpu
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data_dblp(path='../ACM3025.mat'):
    data = sio.loadmat(path)
    truelabels, truefeatures = data['label'], data['feature'].astype(float)
    N = truefeatures.shape[0]
    rownetworks = [data['PAP'] - np.eye(N), data['PLP'] - np.eye(N)]  # , data['PTP'] - np.eye(N)]
    a = rownetworks[1].max()
    y = truelabels
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    train_mask = sample_mask(train_idx, y.shape[0])
    val_mask = sample_mask(val_idx, y.shape[0])
    test_mask = sample_mask(test_idx, y.shape[0])

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    y_train[train_mask, :] = y[train_mask, :]
    y_val[val_mask, :] = y[val_mask, :]
    y_test[test_mask, :] = y[test_mask, :]

    # return selected_idx, selected_idx_2
    print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
                                                                                          y_val.shape,
                                                                                          y_test.shape,
                                                                                          train_idx.shape,
                                                                                          val_idx.shape,
                                                                                          test_idx.shape))
    truefeatures_list = [truefeatures, truefeatures, truefeatures]
    return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask

class HAN():
    def __init__(self, config, data_process):
        self.batch_size = 1
        self.nb_epochs = config.epochs
        self.patience = config.patience
        self.lr = config.alpha  # learning rate
        self.l2_coef = config.lr_decay  # weight decay
        # numbers of hidden units per each attention head in each layer
        self.hid_units = [8]
        self.n_heads = [8, 1]  # additional entry for the output layer
        self.residual = False
        self.nonlinearity = tf.nn.elu
        self.model = HeteGAT_multi
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.checkpt_file = './output/temp/model.ckpt'
        self.featype = config.featype
        self.data_process =data_process
        self.out_emd_file = config.out_emd_file

    def train(self):
        # use adj_list as fea_list, have a try~
        adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = self.data_process.data_process()
        if self.featype == 'adj':
            fea_list = adj_list


        nb_nodes = fea_list[0].shape[0]
        ft_size = fea_list[0].shape[1]
        nb_classes = y_train.shape[1]

        # adj = adj.todense()

        # features = features[np.newaxis]  # [1, nb_node, ft_size]
        fea_list = [fea[np.newaxis] for fea in fea_list]
        adj_list = [adj[np.newaxis] for adj in adj_list]
        y_train = y_train[np.newaxis]
        y_val = y_val[np.newaxis]
        y_test = y_test[np.newaxis]
        train_mask = train_mask[np.newaxis]
        val_mask = val_mask[np.newaxis]
        test_mask = test_mask[np.newaxis]

        biases_list = [adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]

        print('build graph...')
        with tf.Graph().as_default():
            with tf.name_scope('input'):
                ftr_in_list = [tf.placeholder(dtype=tf.float32,
                                              shape=(self.batch_size, nb_nodes, ft_size),
                                              name='ftr_in_{}'.format(i))
                               for i in range(len(fea_list))]
                bias_in_list = [tf.placeholder(dtype=tf.float32,
                                               shape=(self.batch_size, nb_nodes, nb_nodes),
                                               name='bias_in_{}'.format(i))
                                for i in range(len(biases_list))]
                lbl_in = tf.placeholder(dtype=tf.int32, shape=(
                    self.batch_size, nb_nodes, nb_classes), name='lbl_in')
                msk_in = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, nb_nodes),
                                        name='msk_in')
                attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
                ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
                is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')
            # forward
            logits, final_embedding, att_val = self.model.inference(ftr_in_list, nb_classes, nb_nodes, is_train,
                                                               attn_drop, ffd_drop,
                                                               bias_mat_list=bias_in_list,
                                                               hid_units=self.hid_units, n_heads=self.n_heads,
                                                               residual=self.residual, activation=self.nonlinearity)

            # cal masked_loss
            log_resh = tf.reshape(logits, [-1, nb_classes])
            lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
            msk_resh = tf.reshape(msk_in, [-1])
            loss = self.model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
            accuracy = self.model.masked_accuracy(log_resh, lab_resh, msk_resh)
            # optimzie
            train_op = self.model.training(loss, self.lr, self.l2_coef)

            saver = tf.train.Saver()

            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

            vlss_mn = np.inf
            vacc_mx = 0.0
            curr_step = 0

            with tf.Session(config=self.config) as sess:
                sess.run(init_op)

                train_loss_avg = 0
                train_acc_avg = 0
                val_loss_avg = 0
                val_acc_avg = 0

                for epoch in range(self.nb_epochs):
                    tr_step = 0

                    tr_size = fea_list[0].shape[0]
                    # ================   training    ============
                    while tr_step * self.batch_size < tr_size:
                        fd1 = {i: d[tr_step * self.batch_size:(tr_step + 1) * self.batch_size]
                               for i, d in zip(ftr_in_list, fea_list)}
                        fd2 = {i: d[tr_step * self.batch_size:(tr_step + 1) * self.batch_size]
                               for i, d in zip(bias_in_list, biases_list)}
                        fd3 = {lbl_in: y_train[tr_step * self.batch_size:(tr_step + 1) * self.batch_size],
                               msk_in: train_mask[tr_step * self.batch_size:(tr_step + 1) * self.batch_size],
                               is_train: True,
                               attn_drop: 0.6,
                               ffd_drop: 0.6}
                        fd = fd1
                        fd.update(fd2)
                        fd.update(fd3)
                        _, loss_value_tr, acc_tr, att_val_train = sess.run([train_op, loss, accuracy, att_val],
                                                                           feed_dict=fd)
                        train_loss_avg += loss_value_tr
                        train_acc_avg += acc_tr
                        tr_step += 1

                    vl_step = 0
                    vl_size = fea_list[0].shape[0]
                    # =============   val       =================
                    while vl_step * self.batch_size < vl_size:
                        # fd1 = {ftr_in: features[vl_step * batch_size:(vl_step + 1) * batch_size]}
                        fd1 = {i: d[vl_step * self.batch_size:(vl_step + 1) * self.batch_size]
                               for i, d in zip(ftr_in_list, fea_list)}
                        fd2 = {i: d[vl_step * self.batch_size:(vl_step + 1) * self.batch_size]
                               for i, d in zip(bias_in_list, biases_list)}
                        fd3 = {lbl_in: y_val[vl_step * self.batch_size:(vl_step + 1) * self.batch_size],
                               msk_in: val_mask[vl_step * self.batch_size:(vl_step + 1) * self.batch_size],
                               is_train: False,
                               attn_drop: 0.0,
                               ffd_drop: 0.0}

                        fd = fd1
                        fd.update(fd2)
                        fd.update(fd3)
                        loss_value_vl, acc_vl = sess.run([loss, accuracy],
                                                         feed_dict=fd)
                        val_loss_avg += loss_value_vl
                        val_acc_avg += acc_vl
                        vl_step += 1
                    # import pdb; pdb.set_trace()
                    print('Epoch: {}, att_val: {}'.format(epoch, np.mean(att_val_train, axis=0)))
                    print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                          (train_loss_avg / tr_step, train_acc_avg / tr_step,
                           val_loss_avg / vl_step, val_acc_avg / vl_step))

                    if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
                        if val_acc_avg / vl_step >= vacc_mx and val_loss_avg / vl_step <= vlss_mn:
                            vacc_early_model = val_acc_avg / vl_step
                            vlss_early_model = val_loss_avg / vl_step
                            saver.save(sess, self.checkpt_file)
                        vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
                        vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
                        curr_step = 0
                    else:
                        curr_step += 1
                        if curr_step == self.patience:
                            print('Early stop! Min loss: ', vlss_mn,
                                  ', Max accuracy: ', vacc_mx)
                            print('Early stop model validation loss: ',
                                  vlss_early_model, ', accuracy: ', vacc_early_model)
                            break

                    train_loss_avg = 0
                    train_acc_avg = 0
                    val_loss_avg = 0
                    val_acc_avg = 0

                saver.restore(sess, self.checkpt_file)
                print('load model from : {}'.format(self.checkpt_file))
                ts_size = fea_list[0].shape[0]
                ts_step = 0
                ts_loss = 0.0
                ts_acc = 0.0

                while ts_step * self.batch_size < ts_size:
                    # fd1 = {ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size]}
                    fd1 = {i: d[ts_step * self.batch_size:(ts_step + 1) * self.batch_size]
                           for i, d in zip(ftr_in_list, fea_list)}
                    fd2 = {i: d[ts_step * self.batch_size:(ts_step + 1) * self.batch_size]
                           for i, d in zip(bias_in_list, biases_list)}
                    fd3 = {lbl_in: y_test[ts_step * self.batch_size:(ts_step + 1) * self.batch_size],
                           msk_in: test_mask[ts_step * self.batch_size:(ts_step + 1) * self.batch_size],

                           is_train: False,
                           attn_drop: 0.0,
                           ffd_drop: 0.0}

                    fd = fd1
                    fd.update(fd2)
                    fd.update(fd3)
                    loss_value_ts, acc_ts, jhy_final_embedding = sess.run([loss, accuracy, final_embedding],
                                                                          feed_dict=fd)
                    ts_loss += loss_value_ts
                    ts_acc += acc_ts
                    ts_step += 1

                print('Test loss:', ts_loss / ts_step,
                      '; Test accuracy:', ts_acc / ts_step)

                print('start knn, kmean.....')
                xx = np.expand_dims(jhy_final_embedding, axis=0)[test_mask]

                # xx = xx / LA.norm(xx, axis=1)
                yy = y_test[test_mask]

                print('xx: {}, yy: {}'.format(xx.shape, yy.shape))
                write2file(self.out_emd_file, jhy_final_embedding, self.data_process)
                # my_KNN(xx, yy)
                # my_Kmeans(xx, yy)

                sess.close()


def write2file(filename, mtx, data):
    result = {}
    for i in range(len(mtx)):
        node_n = data.s + str(i)
        result[data.s + data.find_dict[node_n]] = mtx[i]
    dim = len(mtx[0])
    write_emd_file(filename, result, dim)


import numpy as np


from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, f1_score



def my_KNN(x, y, k=5, split_list=[0.2, 0.4, 0.6, 0.8], time=10, show_train=True, shuffle=True):
    x = np.array(x)
    x = np.squeeze(x)
    y = np.array(y)
    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    for split in split_list:
        ss = split
        split = int(x.shape[0] * split)
        micro_list = []
        macro_list = []
        if time:
            for i in range(time):
                if shuffle:
                    permutation = np.random.permutation(x.shape[0])
                    x = x[permutation, :]
                    y = y[permutation]
                # x_true = np.array(x_true)
                train_x = x[:split, :]
                test_x = x[split:, :]

                train_y = y[:split]
                test_y = y[split:]

                estimator = KNeighborsClassifier(n_neighbors=k)
                estimator.fit(train_x, train_y)
                y_pred = estimator.predict(test_x)
                f1_macro = f1_score(test_y, y_pred, average='macro')
                f1_micro = f1_score(test_y, y_pred, average='micro')
                macro_list.append(f1_macro)
                micro_list.append(f1_micro)
            print('KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(
                time, ss, k, sum(macro_list) / len(macro_list), sum(micro_list) / len(micro_list)))


def my_Kmeans(x, y, k=4, time=10, return_NMI=False):
    x = np.array(x)
    x = np.squeeze(x)
    y = np.array(y)

    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)

    estimator = KMeans(n_clusters=k)
    ARI_list = []  # adjusted_rand_score(
    NMI_list = []
    if time:
        # print('KMeans exps {}次 æ±~B平å~]~G '.format(time))
        for i in range(time):
            estimator.fit(x, y)
            y_pred = estimator.predict(x)
            score = normalized_mutual_info_score(y, y_pred)
            NMI_list.append(score)
            s2 = adjusted_rand_score(y, y_pred)
            ARI_list.append(s2)
        # print('NMI_list: {}'.format(NMI_list))
        score = sum(NMI_list) / len(NMI_list)
        s2 = sum(ARI_list) / len(ARI_list)
        print('NMI (10 avg): {:.4f} , ARI (10avg): {:.4f}'.format(score, s2))

    else:
        estimator.fit(x, y)
        y_pred = estimator.predict(x)
        score = normalized_mutual_info_score(y, y_pred)
        print("NMI on all label data: {:.5f}".format(score))
    if return_NMI:
        return score, s2

def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)