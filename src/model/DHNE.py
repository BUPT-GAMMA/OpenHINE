import numpy as np
import os, sys
import tensorflow as tf
import argparse
from functools import reduce
import math
import time
import copy
import collections
import scipy.io as sio
import operator
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as s_vstack
import itertools
import random
import json
from keras.models import Model
from keras import regularizers, optimizers
from keras.layers import Input, Dense, concatenate
from keras import backend as K
from keras.models import load_model
from src.utils.utils import *



Datasets = collections.namedtuple('Datasets', ['train', 'test', 'embeddings', 'node_cluster',
                                                'labels', 'idx_label', 'label_name'])

            
class DataSet(object):

    def __init__(self, edge, nums_type, **kwargs):
        self.edge = edge
        self.edge_set = set(map(tuple, edge)) 
        self.nums_type = nums_type
        self.kwargs = kwargs
        self.nums_examples = len(edge)
        self.epochs_completed = 0
        self.index_in_epoch = 0

    def next_batch(self, embeddings, batch_size=16, num_neg_samples=1, pair_radio=0.9, sparse_input=True):
        """
            Return the next `batch_size` examples from this data set.
            if num_neg_samples = 0, there is no negative sampling.
        """
        while 1:
            start = self.index_in_epoch
            self.index_in_epoch += batch_size
            if self.index_in_epoch > self.nums_examples:
                self.epochs_completed += 1
                np.random.shuffle(self.edge)
                start = 0
                self.index_in_epoch = batch_size
                assert self.index_in_epoch <= self.nums_examples
            end = self.index_in_epoch
            neg_data = []
            for i in range(start, end):
                n_neg = 0
                while(n_neg < num_neg_samples):
                    index = copy.deepcopy(self.edge[i])
                    mode = np.random.rand()
                    if mode < pair_radio:
                        type_ = np.random.randint(3)
                        node = np.random.randint(self.nums_type[type_])
                        index[type_] = node
                    else:
                        types_ = np.random.choice(3, 2, replace=False)
                        node_1 = np.random.randint(self.nums_type[types_[0]])
                        node_2 = np.random.randint(self.nums_type[types_[1]])
                        index[types_[0]] = node_1
                        index[types_[1]] = node_2
                    if tuple(index) in self.edge_set:
                        continue
                    n_neg += 1
                    neg_data.append(index)
            if len(neg_data) > 0:
                batch_data = np.vstack((self.edge[start:end], neg_data))
                nums_batch = len(batch_data)
                labels = np.zeros(nums_batch)
                labels[0:end-start] = 1
                perm = np.random.permutation(nums_batch)
                batch_data = batch_data[perm]
                labels = labels[perm]
            else:
                batch_data = self.edge[start:end]
                nums_batch = len(batch_data)
                labels = np.ones(len(batch_data))
            batch_e = embedding_lookup(embeddings, batch_data, sparse_input)
            yield (dict([('input_{}'.format(i), batch_e[i]) for i in range(3)]),
                    dict([('decode_{}'.format(i), batch_e[i]) for i in range(3)]+[('classify_layer', labels)]))

def embedding_lookup(embeddings, index, sparse_input=True):
    if sparse_input:
        return [embeddings[i][index[:, i], :].todense()  for i in range(3)]
    else:
        return [embeddings[i][index[:, i], :]  for i in range(3)]

def read_data_sets(train_dir):
    TRAIN_FILE = 'train_data.npz'
    TEST_FILE = 'test_data.npz'
    data = np.load(os.path.join(str(train_dir), TRAIN_FILE))
    train_data = DataSet(data['train_data'], data['nums_type'])
    labels = data['labels'] if 'labels' in data else None
    idx_label = data['idx_label'] if 'idx_label' in data else None
    label_set = data['label_name'] if 'label_name' in data else None
    del data
    data = np.load(os.path.join(str(train_dir), TEST_FILE))
    test_data = DataSet(data['test_data'], data['nums_type'])
    node_cluster = data['node_cluster'] if 'node_cluster' in data else None
    test_labels = data['labels'] if 'labels' in data else None
    del data
    embeddings = generate_embeddings(train_data.edge, train_data.nums_type)
    return Datasets(train=train_data, test=test_data, embeddings=embeddings, node_cluster=node_cluster,
                labels=labels, idx_label=idx_label, label_name=label_set)

def generate_H(edge, nums_type):
    nums_examples = len(edge)
    H = [csr_matrix((np.ones(nums_examples), (edge[:, i], range(nums_examples))), shape=(nums_type[i], nums_examples)) for i in range(3)]
    return H

def dense_to_onehot(labels):
    return np.array(map(lambda x: [x*0.5+0.5, x*-0.5+0.5], list(labels)), dtype=float)

def generate_embeddings(edge, nums_type, H=None):
    if H is None:
        H = generate_H(edge, nums_type)
    embeddings = [H[i].dot(s_vstack([H[j] for j in range(3) if j != i]).T).astype('float') for i in range(3)]
    for i in range(3):
        col_max = np.array(embeddings[i].max(0).todense()).flatten()
        _, col_index = embeddings[i].nonzero()
        embeddings[i].data /= col_max[col_index]
    return embeddings

class hypergraph(object):
    def __init__(self, dim_feature,embedding_size,hidden_size,learning_rate,alpha,batch_size,num_neg_samples,epochs_to_train,output_embfold,output_modelfold,prefix_path ,reflect):
        self.dim_feature=dim_feature
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.learning_rate=learning_rate
        self.alpha=alpha
        self.batch_size=batch_size
        self.num_neg_samples=num_neg_samples
        self.epochs_to_train=epochs_to_train
        self.output_modelfold=output_modelfold
        self.output_embfold=output_embfold
        self.prefix_path=prefix_path
        self.reflect = reflect
        self.mp = "a-p-s".split("-")
        self.build_model()


    def sparse_autoencoder_error(self, y_true, y_pred):
        return K.mean(K.square(K.sign(y_true)*(y_true-y_pred)), axis=-1)


    def build_model(self):
        self.inputs = [Input(shape=(self.dim_feature[i], ), name='input_{}'.format(i), dtype='float') for i in range(3)]

        self.encodeds = [Dense(self.embedding_size, activation='tanh', name='encode_{}'.format(i))(self.inputs[i]) for i in range(3)]
        self.decodeds = [Dense(self.dim_feature[i], activation='sigmoid', name='decode_{}'.format(i),
                        activity_regularizer = regularizers.l2(0.0))(self.encodeds[i]) for i in range(3)]

        self.merged = concatenate(self.encodeds, axis=1)
        self.hidden_layer = Dense(self.hidden_size, activation='tanh', name='full_connected_layer')(self.merged)
        self.ouput_layer = Dense(1, activation='sigmoid', name='classify_layer')(self.hidden_layer)

        self.model = Model(inputs=self.inputs, outputs=self.decodeds+[self.ouput_layer])

        self.model.compile(optimizer=optimizers.RMSprop(lr=self.learning_rate),
                loss=[self.sparse_autoencoder_error]*3+['binary_crossentropy'],
                              loss_weights=[self.alpha]*3+[1.0],
                              metrics=dict([('decode_{}'.format(i), 'mse') for i in range(3)]+[('classify_layer', 'accuracy')]))

        self.model.summary()

    def train(self, dataset):
        self.hist = self.model.fit_generator(
                dataset.train.next_batch(dataset.embeddings, self.batch_size, num_neg_samples=self.num_neg_samples),
                validation_data=dataset.test.next_batch(dataset.embeddings, self.batch_size, num_neg_samples=self.num_neg_samples),
                validation_steps=1,
                steps_per_epoch=math.ceil(dataset.train.nums_examples/self.batch_size),
                epochs=self.epochs_to_train, verbose=2)

    def predict(self, embeddings, data):
        test = embedding_lookup(embeddings, data)
        return self.model.predict(test, batch_size=self.batch_size)[3]

    def fill_feed_dict(self, embeddings, nums_type, x, y):
        batch_e = embedding_lookup(embeddings, x)
        return (dict([('input_{}'.format(i), batch_e[i]) for i in range(3)]),
                dict([('decode_{}'.format(i), batch_e[i]) for i in range(3)]+[('classify_layer', y)]))
        return res

    def get_embeddings(self, dataset):
        shift = np.append([0], np.cumsum(dataset.train.nums_type))
        embeddings = []
        for i in range(3):
            index = range(dataset.train.nums_type[i])
            batch_num = math.ceil(1.0*len(index)/self.batch_size)
            ls = np.array_split(index, batch_num)
            ps = []
            for j, lss in enumerate(ls):
                embed = K.get_session().run(self.encodeds[i], feed_dict={
                    self.inputs[i]: dataset.embeddings[i][lss, :].todense()})
                ps.append(embed)
            ps = np.vstack(ps)
            embeddings.append(ps)
        return embeddings
    
    def save(self):
        prefix = '{}_{}'.format(str(self.prefix_path), self.embedding_size)
        prefix_path = os.path.join(str(self.output_modelfold), prefix)
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)
        self.model.save(os.path.join(prefix_path, 'model.h5'))
    
    def save_embeddings(self, dataset, file_name='node.txt'):
        path = str(self.output_embfold) + file_name
        # if not os.path.exists(str(self.output_embfold) + "DHNE"):
        #     os.makedirs(str(self.output_embfold) + "DHNE")
        emds = self.get_embeddings(dataset)
        result = {}
        emds[0] = emds[0].tolist()
        emds[1] = emds[1].tolist()
        emds[2] = emds[2].tolist()
        for i, value in self.reflect.items():
            if i[0] == self.mp[0]:
                result[i] = emds[0][value]
            elif i[0] == self.mp[1]:
                result[i] = emds[1][value]
            elif i[0] == self.mp[2]:
                result[i] = emds[2][value]
        dim = len(emds[0][0])
        write_emd_file(path, result, dim)
        # result["ent_embeddings.weight_"+str(0)] = emds[0].tolist()
        # result["ent_embeddings.weight_"+str(1)] = emds[1].tolist()
        # result["ent_embeddings.weight_"+str(2)] = emds[2].tolist()
        # f.write(json.dumps(result))
        # f.close()
    
    def load(self):
        prefix_path = os.path.join(str(self.output_modelfold), '{}_{}'.format(str(self.prefix_path), self.embedding_size))
        self.model = load_model(os.path.join(prefix_path, 'model.h5'), custom_objects={'sparse_autoencoder_error': self.sparse_autoencoder_error})


# def load_hypergraph(data_path):
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     K.set_session(tf.Session(config=config))
#     h = hypergraph(dim_feature,embedding_size,hidden_size,learning_rate,alpha,batch_size,num_neg_samples,epochs_to_train,output_embfold,output_modelfold,prefix_path)
#     h.load()
#     return h


def Process(dataset,dim_feature,embedding_size,hidden_size,learning_rate,alpha,batch_size,num_neg_samples,epochs_to_train,output_embfold,output_modelfold,prefix_path, reflect):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))
    h = hypergraph(dim_feature,embedding_size,hidden_size,learning_rate,alpha,batch_size,num_neg_samples,epochs_to_train,output_embfold,output_modelfold,prefix_path,reflect)
    begin = time.time()
    h.train(dataset)
    end = time.time()
    print("time, ", end-begin)
    # h.save()
    h.save_embeddings(dataset)
    K.clear_session()
