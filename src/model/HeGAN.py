import os
import tensorflow as tf
from src.utils.utils import read_embeddings_with_type,write_emd_file
from src.utils.sampler import Hegan_read_graph
import time
import numpy as np
from src.utils.evaluation import DBLP_evaluation
from src.utils.evaluation import ACM_evaluation


# from src.utils.evaluation  import Aminer_evaluation


class HeGAN():
    def __init__(self, g_hin, args, conf):

        t = time.time()
        print("reading graph...")
        self.n_node, self.n_relation, self.graph = Hegan_read_graph(g_hin)
        self.node_list = list(self.graph.keys())  # range(0, self.n_node)
        print('[%.2f] reading graph finished. #node = %d #relation = %d' % (
            time.time() - t, self.n_node, self.n_relation))

        t = time.time()
        print("read initial embeddings...")
        self.node_embed_init_d = read_embeddings_with_type(filename=conf.pretrain_node_emb_filename,
                                                           n_node=self.n_node,
                                                           n_embed=conf.n_emb, node2id=g_hin.node2id_dict)
        self.node_embed_init_g = read_embeddings_with_type(filename=conf.pretrain_node_emb_filename,
                                                           n_node=self.n_node,
                                                           n_embed=conf.n_emb, node2id=g_hin.node2id_dict)

        # self.rel_embed_init_d = utils.read_embeddings(filename=config.pretrain_rel_emb_filename_d,
        #                                              n_node=self.n_node,
        #                                              n_embed=config.n_emb)
        # self.rel_embed_init_g = utils.read_embeddings(filename=config.pretrain_rel_emb_filename_g,
        #                                              n_node=self.n_node,
        #                                              n_embed=config.n_emb)
        print("[%.2f] read initial embeddings finished." % (time.time() - t))

        print("build GAN model...")
        self.discriminator = None
        self.generator = None
        self.build_generator(conf)
        self.build_discriminator(conf)

        self.latest_checkpoint = tf.train.latest_checkpoint(conf.model_log)
        self.saver = tf.train.Saver()
        if args.dataset == "dblp":
            self.dblp_evaluation = DBLP_evaluation(g_hin.node2id_dict)
        elif args.dataset == "acm":
            self.acm_evaluation = ACM_evaluation(g_hin.node2id_dict)
        # self.aminer_evaluation = Aminer_evaluation()

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init_op)
        self.args = args
        self.show_config(conf)

    def show_config(self, conf):
        config = conf
        print('--------------------')
        print('Model config : ')
        print('dataset = ', self.args.dataset)
        print('batch_size = ', config.batch_size)
        print('lambda_gen = ', config.lambda_gen)
        print('lambda_dis = ', config.lambda_dis)
        print('n_sample = ', config.n_sample)
        print('lr_gen = ', config.lr_gen)
        print('lr_dis = ', config.lr_dis)
        print('n_epoch = ', config.n_epoch)
        print('d_epoch = ', config.d_epoch)
        print('g_epoch = ', config.g_epoch)
        print('n_emb = ', config.n_emb)
        print('sig = ', config.sig)
        print('label smooth = ', config.label_smooth)
        print('--------------------')

    def build_generator(self, conf):
        # with tf.variable_scope("generator"):
        self.generator = Generator(n_node=self.n_node,
                                   n_relation=self.n_relation,
                                   node_emd_init=self.node_embed_init_g,
                                   relation_emd_init=None, config=conf)

    def build_discriminator(self, conf):
        # with tf.variable_scope("discriminator"):
        self.discriminator = Discriminator(n_node=self.n_node,
                                           n_relation=self.n_relation,
                                           node_emd_init=self.node_embed_init_d,
                                           relation_emd_init=None, config=conf)

    def train(self, conf, node2dict):
        config = conf
        print('start traning...')
        for epoch in range(config.n_epoch):
            print('epoch %d' % epoch)
            t = time.time()

            one_epoch_gen_loss = 0.0
            one_epoch_dis_loss = 0.0
            one_epoch_batch_num = 0.0

            # D-step
            # t1 = time.time()
            for d_epoch in range(config.d_epoch):
                np.random.shuffle(self.node_list)
                one_epoch_dis_loss = 0.0
                one_epoch_pos_loss = 0.0
                one_epoch_neg_loss_1 = 0.0
                one_epoch_neg_loss_2 = 0.0

                for index in range(int(len(self.node_list) / config.batch_size)):
                    # t1 = time.time()
                    pos_node_ids, pos_relation_ids, pos_node_neighbor_ids, neg_node_ids_1, neg_relation_ids_1, neg_node_neighbor_ids_1, neg_node_ids_2, neg_relation_ids_2, node_fake_neighbor_embedding = self.prepare_data_for_d(
                        index, config)
                    # t2 = time.time()
                    # print t2 - t1
                    _, dis_loss, pos_loss, neg_loss_1, neg_loss_2 = self.sess.run(
                        [self.discriminator.d_updates, self.discriminator.loss, self.discriminator.pos_loss,
                         self.discriminator.neg_loss_1, self.discriminator.neg_loss_2],
                        feed_dict={self.discriminator.pos_node_id: np.array(pos_node_ids),
                                   self.discriminator.pos_relation_id: np.array(pos_relation_ids),
                                   self.discriminator.pos_node_neighbor_id: np.array(pos_node_neighbor_ids),
                                   self.discriminator.neg_node_id_1: np.array(neg_node_ids_1),
                                   self.discriminator.neg_relation_id_1: np.array(neg_relation_ids_1),
                                   self.discriminator.neg_node_neighbor_id_1: np.array(neg_node_neighbor_ids_1),
                                   self.discriminator.neg_node_id_2: np.array(neg_node_ids_2),
                                   self.discriminator.neg_relation_id_2: np.array(neg_relation_ids_2),
                                   self.discriminator.node_fake_neighbor_embedding: np.array(
                                       node_fake_neighbor_embedding)})

                    one_epoch_dis_loss += dis_loss
                    one_epoch_pos_loss += pos_loss
                    one_epoch_neg_loss_1 += neg_loss_1
                    one_epoch_neg_loss_2 += neg_loss_2

            # G-step

            for g_epoch in range(config.g_epoch):
                np.random.shuffle(self.node_list)
                one_epoch_gen_loss = 0.0

                for index in range(int(len(self.node_list) / config.batch_size)):
                    gen_node_ids, gen_relation_ids, gen_noise_embedding, gen_dis_node_embedding, gen_dis_relation_embedding = self.prepare_data_for_g(
                        index, config)
                    t2 = time.time()

                    _, gen_loss = self.sess.run([self.generator.g_updates, self.generator.loss],
                                                feed_dict={self.generator.node_id: np.array(gen_node_ids),
                                                           self.generator.relation_id: np.array(gen_relation_ids),
                                                           self.generator.noise_embedding: np.array(
                                                               gen_noise_embedding),
                                                           self.generator.dis_node_embedding: np.array(
                                                               gen_dis_node_embedding),
                                                           self.generator.dis_relation_embedding: np.array(
                                                               gen_dis_relation_embedding)})

                    one_epoch_gen_loss += gen_loss

            one_epoch_batch_num = len(self.node_list) / config.batch_size

            # print t2 - t1
            # exit()
            print('[%.2f] gen loss = %.4f, dis loss = %.4f pos loss = %.4f neg loss-1 = %.4f neg loss-2 = %.4f' % \
                  (time.time() - t, one_epoch_gen_loss / one_epoch_batch_num, one_epoch_dis_loss / one_epoch_batch_num,
                   one_epoch_pos_loss / one_epoch_batch_num, one_epoch_neg_loss_1 / one_epoch_batch_num,
                   one_epoch_neg_loss_2 / one_epoch_batch_num))

            if self.args.dataset == 'dblp':
                gen_nmi, dis_nmi = self.evaluate_author_cluster()
                print('Gen NMI score = %.4f Dis NMI score = %.4f' % (gen_nmi, dis_nmi))
                micro_f1s, macro_f1s = self.evaluate_author_classification()
                print('Gen Micro_f1 = %.4f Dis Micro_f1 = %.4f' %(micro_f1s[0], micro_f1s[1]))
                print('Gen Macro_f1 = %.4f Dis Macro_f1 = %.4f' %(macro_f1s[0], macro_f1s[1]))
            # elif args.dataset == 'yelp':
            #     gen_nmi, dis_nmi = self.evaluate_business_cluster()
            #     print('Gen NMI score = %.4f Dis NMI score = %.4f' % (gen_nmi, dis_nmi))
            # micro_f1s, macro_f1s = self.evaluate_business_classification()
            # print 'Gen Micro_f1 = %.4f Dis Micro_f1 = %.4f' %(micro_f1s[0], micro_f1s[1])
            # print 'Gen Macro_f1 = %.4f Dis Macro_f1 = %.4f' %(macro_f1s[0], macro_f1s[1])
            elif self.args.dataset == 'acm':
                gen_nmi, dis_nmi = self.evaluate_paper_cluster()
                print('Gen NMI score = %.4f Dis NMI score = %.4f' % (gen_nmi, dis_nmi))
                micro_f1s, macro_f1s = self.evaluate_paper_classification()
                print('Gen Micro_f1 = %.4f Dis Micro_f1 = %.4f' %(micro_f1s[0], micro_f1s[1]))
                print('Gen Macro_f1 = %.4f Dis Macro_f1 = %.4f' %(macro_f1s[0], macro_f1s[1]))

            # self.evaluate_aminer_link_prediction()
            self.write_embeddings_to_file(config, node2dict)
            # os.system('python ../evaluation/lp_evaluation_2.py')

        print("training completes")

    def prepare_data_for_d(self, index, config):

        pos_node_ids = []
        pos_relation_ids = []
        pos_node_neighbor_ids = []

        # real node and wrong relation
        neg_node_ids_1 = []
        neg_relation_ids_1 = []
        neg_node_neighbor_ids_1 = []

        # fake node and true relation
        neg_node_ids_2 = []
        neg_relation_ids_2 = []
        node_fake_neighbor_embedding = None

        for node_id in self.node_list[index * config.batch_size: (index + 1) * config.batch_size]:
            for i in range(config.n_sample):

                # sample real node and true relation
                relations = list(self.graph[node_id].keys())
                relation_id = relations[np.random.randint(0, len(relations))]
                neighbors = self.graph[node_id][relation_id]
                node_neighbor_id = neighbors[np.random.randint(0, len(neighbors))]

                pos_node_ids.append(node_id)
                pos_relation_ids.append(relation_id)
                pos_node_neighbor_ids.append(node_neighbor_id)

                # sample real node and wrong relation
                neg_node_ids_1.append(node_id)
                neg_node_neighbor_ids_1.append(node_neighbor_id)
                neg_relation_id_1 = np.random.randint(0, self.n_relation)
                while neg_relation_id_1 == relation_id:
                    neg_relation_id_1 = np.random.randint(0, self.n_relation)
                neg_relation_ids_1.append(neg_relation_id_1)

                # sample fake node and true relation
                neg_node_ids_2.append(node_id)
                neg_relation_ids_2.append(relation_id)

        # generate fake node
        noise_embedding = np.random.normal(0.0, config.sig, (len(neg_node_ids_2), config.n_emb))

        node_fake_neighbor_embedding = self.sess.run(self.generator.node_neighbor_embedding,
                                                     feed_dict={self.generator.node_id: np.array(neg_node_ids_2),
                                                                self.generator.relation_id: np.array(
                                                                    neg_relation_ids_2),
                                                                self.generator.noise_embedding: np.array(
                                                                    noise_embedding)})

        return pos_node_ids, pos_relation_ids, pos_node_neighbor_ids, neg_node_ids_1, neg_relation_ids_1, \
               neg_node_neighbor_ids_1, neg_node_ids_2, neg_relation_ids_2, node_fake_neighbor_embedding

    def prepare_data_for_g(self, index, config):
        node_ids = []
        relation_ids = []

        for node_id in self.node_list[index * config.batch_size: (index + 1) * config.batch_size]:
            for i in range(config.n_sample):
                relations = list(self.graph[node_id].keys())
                relation_id = relations[np.random.randint(0, len(relations))]

                node_ids.append(node_id)
                relation_ids.append(relation_id)

        noise_embedding = np.random.normal(0.0, config.sig, (len(node_ids), config.n_emb))

        dis_node_embedding, dis_relation_embedding = self.sess.run(
            [self.discriminator.pos_node_embedding, self.discriminator.pos_relation_embedding],
            feed_dict={self.discriminator.pos_node_id: np.array(node_ids),
                       self.discriminator.pos_relation_id: np.array(relation_ids)})
        return node_ids, relation_ids, noise_embedding, dis_node_embedding, dis_relation_embedding

    def evaluate_author_cluster(self):
        modes = [self.generator, self.discriminator]
        scores = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            score = self.dblp_evaluation.evaluate_author_cluster(embedding_matrix)
            scores.append(score)

        return scores

    def evaluate_author_classification(self):
        modes = [self.generator, self.discriminator]
        micro_f1s = []
        macro_f1s = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            micro_f1, macro_f1 = self.dblp_evaluation.evaluate_author_classification(embedding_matrix)
            micro_f1s.append(micro_f1)
            macro_f1s.append(macro_f1)
        return micro_f1s, macro_f1s

    def evaluate_paper_cluster(self):
        modes = [self.generator, self.discriminator]
        scores = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            score = self.acm_evaluation.evaluate_paper_cluster(embedding_matrix)
            scores.append(score)

        return scores

    def evaluate_paper_classification(self):
        modes = [self.generator, self.discriminator]
        micro_f1s = []
        macro_f1s = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            micro_f1, macro_f1 = self.acm_evaluation.evaluate_paper_classification(embedding_matrix)
            micro_f1s.append(micro_f1)
            macro_f1s.append(macro_f1)
        return micro_f1s, macro_f1s

    # def evaluate_business_cluster(self):
    #     modes = [self.generator, self.discriminator]
    #     scores = []
    #     for i in range(2):
    #         embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
    #         score = self.yelp_evaluation.evaluate_business_cluster(embedding_matrix)
    #         scores.append(score)
    #
    #     return scores
    #
    # def evaluate_business_classification(self):
    #     modes = [self.generator, self.discriminator]
    #     micro_f1s = []
    #     macro_f1s = []
    #     for i in range(2):
    #         embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
    #         micro_f1, macro_f1 = self.yelp_evaluation.evaluate_business_classification(embedding_matrix)
    #         micro_f1s.append(micro_f1)
    #         macro_f1s.append(macro_f1)
    #     return micro_f1s, macro_f1s
    #
    # def evaluate_yelp_link_prediction(self):
    #     modes = [self.generator, self.discriminator]
    #
    #     for i in range(2):
    #         embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
    #
    #         # score = self.yelp_evaluation.evaluate_business_cluster(embedding_matrix)
    #         # print '%d nmi = %.4f' % (i, score)
    #
    #         auc, f1, acc = self.yelp_evaluation.evaluation_link_prediction(embedding_matrix)
    #
    #         print('auc = %.4f f1 = %.4f acc = %.4f' % (auc, f1, acc))

    # def evaluate_dblp_link_prediction(self):
    #     modes = [self.generator, self.discriminator]
    #
    #     for i in range(2):
    #         embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
    #         # relation_matrix = self.sess.run(modes[i].relation_embedding_matrix)
    #
    #         auc, f1, acc = self.dblp_evaluation.evaluation_link_prediction(embedding_matrix)
    #
    #         print('auc = %.4f f1 = %.4f acc = %.4f' % (auc, f1, acc))

    def write_embeddings_to_file(self, config, node2dict):
        modes = [self.generator, self.discriminator]
        m = ['_gen.emb', '_dis.emb']
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)

            embedding_dict = {}
            for node in node2dict:
                embedding_dict[node] = embedding_matrix[node2dict[node]]
            dim = len(embedding_matrix[0])
            file = config.emb_filenames + self.args.dataset + m[i]
            write_emd_file(file, embedding_dict, dim)



class Discriminator():
    def __init__(self, n_node, n_relation, node_emd_init, relation_emd_init, config):
        self.n_node = n_node
        self.n_relation = n_relation
        self.node_emd_init = node_emd_init
        self.relation_emd_init = relation_emd_init
        self.emd_dim = node_emd_init.shape[1]

        # with tf.variable_scope('disciminator'):
        self.node_embedding_matrix = tf.get_variable(name='dis_node_embedding',
                                                     shape=self.node_emd_init.shape,
                                                     initializer=tf.constant_initializer(self.node_emd_init),
                                                     trainable=True)
        self.relation_embedding_matrix = tf.get_variable(name='dis_relation_embedding',
                                                         shape=[self.n_relation, self.emd_dim, self.emd_dim],
                                                         initializer=tf.contrib.layers.xavier_initializer(
                                                             uniform=False),
                                                         trainable=True)

        self.pos_node_id = tf.placeholder(tf.int32, shape=[None])
        self.pos_relation_id = tf.placeholder(tf.int32, shape=[None])
        self.pos_node_neighbor_id = tf.placeholder(tf.int32, shape=[None])

        self.neg_node_id_1 = tf.placeholder(tf.int32, shape=[None])
        self.neg_relation_id_1 = tf.placeholder(tf.int32, shape=[None])
        self.neg_node_neighbor_id_1 = tf.placeholder(tf.int32, shape=[None])

        self.neg_node_id_2 = tf.placeholder(tf.int32, shape=[None])
        self.neg_relation_id_2 = tf.placeholder(tf.int32, shape=[None])
        self.node_fake_neighbor_embedding = tf.placeholder(tf.float32, shape=[None, self.emd_dim])

        self.pos_node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, self.pos_node_id)
        self.pos_node_neighbor_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, self.pos_node_neighbor_id)
        self.pos_relation_embedding = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.pos_relation_id)

        self.neg_node_embedding_1 = tf.nn.embedding_lookup(self.node_embedding_matrix, self.neg_node_id_1)
        self.neg_node_neighbor_embedding_1 = tf.nn.embedding_lookup(self.node_embedding_matrix,
                                                                    self.neg_node_neighbor_id_1)
        self.neg_relation_embedding_1 = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.neg_relation_id_1)

        self.neg_node_embedding_2 = tf.nn.embedding_lookup(self.node_embedding_matrix, self.neg_node_id_2)
        self.neg_relation_embedding_2 = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.neg_relation_id_2)

        # pos loss
        t = tf.reshape(tf.matmul(tf.expand_dims(self.pos_node_embedding, 1), self.pos_relation_embedding),
                       [-1, self.emd_dim])
        self.pos_score = tf.reduce_sum(tf.multiply(t, self.pos_node_neighbor_embedding), axis=1)
        self.pos_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.pos_score), logits=self.pos_score))

        # neg loss_1
        t = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_embedding_1, 1), self.neg_relation_embedding_1),
                       [-1, self.emd_dim])
        self.neg_score_1 = tf.reduce_sum(tf.multiply(t, self.neg_node_neighbor_embedding_1), axis=1)
        self.neg_loss_1 = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.neg_score_1), logits=self.neg_score_1))

        # neg loss_2
        t = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_embedding_2, 1), self.neg_relation_embedding_2),
                       [-1, self.emd_dim])
        self.neg_score_2 = tf.reduce_sum(tf.multiply(t, self.node_fake_neighbor_embedding), axis=1)
        self.neg_loss_2 = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.neg_score_2), logits=self.neg_score_2))

        self.loss = self.pos_loss + self.neg_loss_1 + self.neg_loss_2

        optimizer = tf.train.AdamOptimizer(config.lr_dis)
        # optimizer = tf.train.GradientDescentOptimizer(config.lr_dis)
        # optimizer = tf.train.RMSPropOptimizer(config.lr_dis)
        self.d_updates = optimizer.minimize(self.loss)
        # self.reward = tf.log(1 + tf.exp(tf.clip_by_value(self.score, clip_value_min=-10, clip_value_max=10)))


class Generator():
    def __init__(self, n_node, n_relation, node_emd_init, relation_emd_init, config):
        self.n_node = n_node
        self.n_relation = n_relation
        self.node_emd_init = node_emd_init
        self.relation_emd_init = relation_emd_init
        self.emd_dim = node_emd_init.shape[1]

        # with tf.variable_scope('generator'):
        self.node_embedding_matrix = tf.get_variable(name="gen_node_embedding",
                                                     shape=self.node_emd_init.shape,
                                                     initializer=tf.constant_initializer(self.node_emd_init),
                                                     trainable=True)
        self.relation_embedding_matrix = tf.get_variable(name="gen_relation_embedding",
                                                         shape=[self.n_relation, self.emd_dim, self.emd_dim],
                                                         initializer=tf.contrib.layers.xavier_initializer(
                                                             uniform=False),
                                                         trainable=True)

        self.gen_w_1 = tf.get_variable(name='gen_w',
                                       shape=[self.emd_dim, self.emd_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        self.gen_b_1 = tf.get_variable(name='gen_b',
                                       shape=[self.emd_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        self.gen_w_2 = tf.get_variable(name='gen_w_2',
                                       shape=[self.emd_dim, self.emd_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        self.gen_b_2 = tf.get_variable(name='gen_b_2',
                                       shape=[self.emd_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        # self.bias_vector = tf.Variable(tf.zeros([self.n_node]))

        self.node_id = tf.placeholder(tf.int32, shape=[None])
        self.relation_id = tf.placeholder(tf.int32, shape=[None])
        self.noise_embedding = tf.placeholder(tf.float32, shape=[None, self.emd_dim])

        self.dis_node_embedding = tf.placeholder(tf.float32, shape=[None, self.emd_dim])
        self.dis_relation_embedding = tf.placeholder(tf.float32, shape=[None, self.emd_dim, self.emd_dim])

        self.node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, self.node_id)
        self.relation_embedding = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.relation_id)
        self.node_neighbor_embedding = self.generate_node(self.node_embedding, self.relation_embedding,
                                                          self.noise_embedding)

        t = tf.reshape(tf.matmul(tf.expand_dims(self.dis_node_embedding, 1), self.dis_relation_embedding),
                       [-1, self.emd_dim])
        self.score = tf.reduce_sum(tf.multiply(t, self.node_neighbor_embedding), axis=1)

        self.loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.score) * (1.0 - config.label_smooth),
                                                    logits=self.score)) \
                    + config.lambda_gen * (tf.nn.l2_loss(self.node_embedding) + tf.nn.l2_loss(
            self.relation_embedding) + tf.nn.l2_loss(self.gen_w_1))

        optimizer = tf.train.AdamOptimizer(config.lr_gen)
        # optimizer = tf.train.RMSPropOptimizer(config.lr_gen)
        self.g_updates = optimizer.minimize(self.loss)

    def generate_node(self, node_embedding, relation_embedding, noise_embedding):
        # node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, node_id)
        # relation_embedding = tf.nn.embedding_lookup(self.relation_embedding_matrix, relation_id)

        input = tf.reshape(tf.matmul(tf.expand_dims(node_embedding, 1), relation_embedding), [-1, self.emd_dim])
        # input = tf.concat([input, noise_embedding], axis = 1)
        input = input + noise_embedding

        output = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_1) + self.gen_b_1)
        # input = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_1) + self.gen_b_1)# +  relation_embedding
        # output = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_2) + self.gen_b_2)
        # output = node_embedding + relation_embedding + noise_embedding

        return output
