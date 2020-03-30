from __future__ import division
import tensorflow as tf
import numpy as np
import json
import os
import random

class MG2vecDataProcess(object):
    def __init__(self, random_walk_txt, window_size):
        index2token, token2index, word_and_counts, index2frequency, node_context_pairs = self.parse_random_walk_txt(
            random_walk_txt, window_size)
        self.window_size = window_size
        self.nodeid2index = token2index
        self.index2nodeid = index2token
        self.index2frequency = index2frequency
        index2type, type2indices = self.parse_node_type_mapping_txt(
            self.nodeid2index)
        self.index2type = index2type
        self.type2indices = type2indices
        self.node_context_pairs = node_context_pairs
        self.prepare_sampling_dist(index2frequency, index2type, type2indices)
        self.shuffle()
        self.count = 0
        self.epoch = 1

    def parse_node_type_mapping_txt(self, nodeid2index):
        # this method does not modify any class variables
        index2type = {}
        for key, value in nodeid2index.items():
            index2type[value] = key[0]

        type2indices = {}
        all_types = set(index2type.values())
        for node_type in all_types:
            type2indices[node_type] = []

        for node_index, node_type in index2type.items():
            type2indices[node_type].append(node_index)

        # make array because it will be used with numpy later
        for node_type in all_types:
            type2indices[node_type] = np.array(type2indices[node_type])

        return index2type, type2indices

    def parse_random_walk_txt(self, random_walk_txt, window_size):
        # this method does not modify any class variables
        # this will NOT make any <UKN> so don't use for NLP.
        word_and_counts = {}
        with open(random_walk_txt) as f:
            for line in f:
                sent = [word.strip() for word in line.strip().split(' ')]
                for word in sent:
                    if len(word) == 0:
                        continue
                    if word in word_and_counts:
                        word_and_counts[word] += 1
                    else:
                        word_and_counts[word] = 1

        print("The number of unique words:%d" % len(word_and_counts))
        index2token = dict((i, word)
                           for i, word in enumerate(word_and_counts.keys()))
        print(index2token)
        token2index = dict((v, k) for k, v in index2token.items())
        index2frequency = dict(
            (token2index[word],
             freq) for word,
            freq in word_and_counts.items())
        # print(index2frequency)
        # word_word = scipy.sparse.lil_matrix((len(token2index), len(token2index)), dtype=np.int32)
        node_context_pairs = []  # let's use naive way now

        print("window size %d" % window_size)

        with open(random_walk_txt) as f:
            for line in f:
                sent = [token2index[word.strip()] for word in line.split(
                    ' ') if word.strip() in token2index]
                sent_length = len(sent)
                for target_word_position, target_word_idx in enumerate(sent):
                    start = max(0, target_word_position - window_size)
                    end = min(
                        sent_length,
                        target_word_position +
                        window_size +
                        1)
                    context = sent[start:target_word_position] + \
                        sent[target_word_position + 1:end + 1]
                    for contex_word_idx in context:
                        node_context_pairs.append(
                            (target_word_idx, contex_word_idx))
                        # word_word[target_word_idx,contex_word_idx]+=1
        # word_word=word_word.tocsr()
        # word_and_counts means token2frequency
        return index2token, token2index, word_and_counts, index2frequency, node_context_pairs

    def get_one_batch(self):
        if self.count == len(self.node_context_pairs):
            self.count = 0
            self.epoch += 1
        node_context_pair = self.node_context_pairs[self.count]
        self.count += 1
        return node_context_pair

    def get_batch(self, batch_size):
        pairs = np.array([self.get_one_batch() for i in range(batch_size)])
        return pairs[:, 0], pairs[:, 1]

    def shuffle(self):
        random.shuffle(self.node_context_pairs)

    def get_negative_samples(self, pos_index, num_negatives, care_type):
        # if care_type is True it's a heterogeneous negative sampling
        # same output format as
        # https://www.tensorflow.org/api_docs/python/tf/nn/log_uniform_candidate_sampler
        pos_prob = self.sampling_prob[pos_index]
        if not care_type:
            negative_samples = np.random.choice(
                len(self.index2nodeid), size=num_negatives, replace=False, p=self.sampling_prob)
            negative_probs = self.sampling_prob[negative_samples]
        else:
            node_type = self.index2type[pos_index]
            sampling_probs = self.type2probs[node_type]
            sampling_candidates = self.type2indices[node_type]
            negative_samples_indices = np.random.choice(
                len(sampling_candidates), size=num_negatives, replace=False, p=sampling_probs)

            negative_samples = sampling_candidates[negative_samples_indices]
            negative_probs = sampling_probs[negative_samples_indices]

        # print(negative_samples,pos_prob,negative_probs)
        return negative_samples, pos_prob.reshape((-1, 1)), negative_probs

    def prepare_sampling_dist(self, index2frequency, index2type, type2indices):
        sampling_prob = np.zeros(len(index2frequency))
        for i in range(len(index2frequency)):
            sampling_prob[i] = index2frequency[i]
        # from
        # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        sampling_prob = sampling_prob**(3.0 / 4.0)

        # normalize the distributions
        # for caring type
        all_types = set(index2type.values())
        type2probs = {}
        for node_type in all_types:
            indicies_for_a_type = type2indices[node_type]
            type2probs[node_type] = np.array(
                sampling_prob[indicies_for_a_type])
            type2probs[node_type] = type2probs[node_type] / \
                np.sum(type2probs[node_type])

        # if not caring type
        sampling_prob = sampling_prob / np.sum(sampling_prob)

        self.sampling_prob = sampling_prob
        self.type2probs = type2probs


def build_model(BATCH_SIZE, VOCAB_SIZE, EMBED_SIZE, NUM_SAMPLED):
    '''
    Build the model (i.e. computational graph) and return the placeholders (input and output) and the loss
    '''
    # define the placeholders for input and output
    with tf.name_scope('data'):
        center_node = tf.placeholder(
            tf.int32, shape=[BATCH_SIZE], name='center_node')
        context_node = tf.placeholder(
            tf.int32, shape=[
                BATCH_SIZE, 1], name='context_node')
        negative_samples = (tf.placeholder(tf.int32, shape=[NUM_SAMPLED], name='negative_samples'),
                            tf.placeholder(
            tf.float32, shape=[
                BATCH_SIZE, 1], name='true_expected_count'),
            tf.placeholder(tf.float32, shape=[NUM_SAMPLED], name='sampled_expected_count'))

    # https://github.com/tensorflow/tensorflow/blob/624bcfe409601910951789325f0b97f520c0b1ee/tensorflow/python/ops/nn_impl.py#L943-L946
    # Sample the negative labels.
    #   sampled shape: [num_sampled] tensor
    #   true_expected_count shape = [batch_size, 1] tensor
    #   sampled_expected_count shape = [num_sampled] tensor

    # Assemble this part of the graph on the CPU. You can change it to GPU if you have GPU
    # define weights. In word2vec, it's actually the weights that we care about

    with tf.name_scope('embedding_matrix'):
        embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0),
                                   name='embed_matrix')

    # define the inference
    with tf.name_scope('loss'):
        embed = tf.nn.embedding_lookup(embed_matrix, center_node, name='embed')

        # construct variables for NCE loss
        nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE],
                                                     stddev=1.0 / (EMBED_SIZE ** 0.5)),
                                 name='nce_weight')
        nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')

        # define loss function to be NCE loss function
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                             biases=nce_bias,
                                             labels=context_node,
                                             inputs=embed,
                                             sampled_values=negative_samples,
                                             num_sampled=NUM_SAMPLED,
                                             num_classes=VOCAB_SIZE), name='loss')

        loss_summary = tf.summary.scalar("loss_summary", loss)

    return center_node, context_node, negative_samples, loss


def traning_op(loss, LEARNING_RATE):
    '''
    Return optimizer
    define one step for SGD
    '''
    # define optimizer
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    return optimizer


def train(center_node_placeholder, context_node_placeholder, negative_samples_placeholder, loss, dataset,
          optimizer, NUM_EPOCHS, BATCH_SIZE, NUM_SAMPLED, care_type, LOG_DIRECTORY, LOG_INTERVAL, MAX_KEEP_MODEL):
    '''
    tensorflow training loop
    define SGD training
    *epoch index starts from 1! not 0.
    '''
    care_type = True if care_type == 1 else False

    # For tensorboard
    merged = tf.summary.merge_all()
    # Add ops to save and restore all the variables.
    # tf.train.Saver(max_to_keep=100)
    saver = tf.train.Saver(max_to_keep=MAX_KEEP_MODEL)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_loss = 0.0  # we use this to calculate late average loss in the last LOG_INTERVAL steps
        writer = tf.summary.FileWriter(LOG_DIRECTORY, sess.graph)
        global_iteration = 0
        iteration = 0
        while (dataset.epoch <= NUM_EPOCHS):
            print("s")
            current_epoch = dataset.epoch
            center_node_batch, context_node_batch = dataset.get_batch(
                BATCH_SIZE)
            negative_samples = dataset.get_negative_samples(
                pos_index=context_node_batch[0], num_negatives=NUM_SAMPLED, care_type=care_type)
            context_node_batch = context_node_batch.reshape((-1, 1))
            loss_batch, _, summary_str = sess.run([loss, optimizer, merged],
                                                  feed_dict={
                center_node_placeholder: center_node_batch,
                context_node_placeholder: context_node_batch,
                negative_samples_placeholder: negative_samples
            })
            writer.add_summary(summary_str, global_iteration)
            total_loss += loss_batch

            # print(loss_batch)

            iteration += 1
            global_iteration += 1

            if LOG_INTERVAL > 0:
                if global_iteration % LOG_INTERVAL == 0:
                    print(
                        'Average loss: {:5.1f}'.format(
                            total_loss / LOG_INTERVAL))
                    total_loss = 0.0
                    # save model
                    model_path = os.path.join(
                        LOG_DIRECTORY, "/model_temp.ckpt")
                    save_path = saver.save(sess, model_path)
                    print("Model saved in file: %s" % save_path)

            if dataset.epoch - current_epoch > 0:
                print("Epoch %d end" % current_epoch)
                dataset.shuffle()
                # save model
                model_path = os.path.join(
                    LOG_DIRECTORY,
                    "model_epoch%d.ckpt" %
                    dataset.epoch)
                save_path = saver.save(sess, model_path)
                print("Model saved in file: %s" % save_path)
                print(
                    'Average loss in this epoch: {:5.1f}'.format(
                        total_loss / iteration))
                total_loss = 0.0
                iteration = 0

        model_path = os.path.join(LOG_DIRECTORY, "model_final.ckpt")
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)
        writer.close()

        print("Save final embeddings as numpy array")
        np_node_embeddings = tf.get_default_graph().get_tensor_by_name(
            "embedding_matrix/embed_matrix:0")
        np_node_embeddings = sess.run(np_node_embeddings)
        np.savez(
            os.path.join(
                LOG_DIRECTORY,
                "node_embeddings.npz"),
            np_node_embeddings)

        with open(os.path.join(LOG_DIRECTORY, "index2nodeid.json"), 'w') as f:
            json.dump(dataset.index2nodeid, f, sort_keys=True, indent=4)
