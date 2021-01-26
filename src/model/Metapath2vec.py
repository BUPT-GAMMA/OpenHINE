import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import init
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression




class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, inputFileName, min_count, care_type):

        self.negatives = []
        self.discards = []
        self.negpos = 0
        self.care_type = care_type
        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.inputFileName = inputFileName
        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName):
            line = line.split()
        # for line in data.split('\n'):
        #     line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        for w, c in word_frequency.items():
            # if c < min_count:
            #     continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1

        self.word_count = len(self.word2id)
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        # get a frequency table for sub-sampling. Note that the frequency is adjusted by
        # sub-sampling tricks.
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        # get a table for negative sampling, if word with index 2 appears twice, then 2 will be listed
        # in the table twice.
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)
        self.sampling_prob = ratio

    def getNegatives(self, target, size):  # TODO check equality with target
        if self.care_type == 0:
            response = self.negatives[self.negpos:self.negpos + size]
            self.negpos = (self.negpos + size) % len(self.negatives)
            if len(response) != size:
                return np.concatenate((response, self.negatives[0:self.negpos]))
        return response


# -----------------------------------------------------------------------------------------------------------------

class Metapath2vecDataset(Dataset):
    def __init__(self, data, window_size, neg_num):
        # read in dataset, window_size and input filename
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.inputFileName, encoding="ISO-8859-1")
        self.neg_num = neg_num

    def __len__(self):
        # return the number of walks
        return self.data.sentences_count

    def __getitem__(self, idx):
        # return the list of pairs (center, context, 5 negatives)
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()
            if len(line) > 1:
                words = line.split()
                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]
                    # and np.random.rand() < self.dataset.discards[self.dataset.word2id[w]]
                    pair_catch = []
                    for i, u in enumerate(word_ids):
                        for j, v in enumerate(
                                word_ids[max(i - self.window_size, 0):i] + word_ids[i + 1:i + self.window_size]):
                            # for j, v in enumerate(
                            #
                            #         word_ids[max(i - self.window_size, 0):i + self.window_size]):

                            # for j, v in enumerate(
                            #
                            #         word_ids[max(i - self.window_size, 0):min(len(word_ids) - 1,i + self.window_size)]):
                            # end = min[len(word_ids) - 1,i + self.window_size]
                            assert u < self.data.word_count
                            assert v < self.data.word_count
                            # if u == v:
                            #
                            #     continue
                            pair_catch.append((u, v, self.data.getNegatives(v, self.neg_num)))
                    return pair_catch

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)


"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""


class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))

    def eva_embedding(self, word2id, label, n_label):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        embedding_dict = {}
        for node in label:
            if(word2id.get(node,-1) == -1):
                print(node)
            embedding_dict[node] = embedding[word2id[node]].tolist()

        NMI = 0
        mi_all = 0
        ma_all = 0
        n = 1
        for i in range(n):
            NMI = NMI + self.evaluate_cluster(embedding_dict, label, n_label)
            micro_f1, macro_f1 = self.evaluate_clf(embedding_dict, label)
            mi_all += micro_f1
            ma_all += macro_f1
        NMI = NMI / n
        micro_f1 = mi_all / n
        macro_f1 = ma_all / n
        print('NMI = %.4f' % NMI)
        print('Micro_F1 = %.4f, Macro_F1 = %.4f' % (micro_f1, macro_f1))

    def evaluate_cluster(self, embedding_dict, label, n_label):
        X = []
        Y = []
        for p in label:
            X.append(embedding_dict[p])
            Y.append(label[p])

        Y_pred = KMeans(n_label, random_state=0).fit(np.array(X)).predict(X)
        nmi =  normalized_mutual_info_score(np.array(Y), Y_pred)
        return nmi

    def evaluate_clf(self, embedding_dict, label):
        X = []
        Y = []
        for p in label:
            X.append(embedding_dict[p])
            Y.append(label[p])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        LR = LogisticRegression()
        LR.fit(X_train, Y_train)
        Y_pred = LR.predict(X_test)

        micro_f1 = f1_score(Y_test, Y_pred, average = 'micro')
        macro_f1 = f1_score(Y_test, Y_pred, average = 'macro')
        return micro_f1, macro_f1

class Metapath2VecTrainer:

    def __init__(self, args, g_hin):
        self.data = DataReader(args.temp_file, 0, 0)# min_cont & care_type

        dataset = Metapath2vecDataset(self.data, args.window_size, args.neg_num)

        self.dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers, collate_fn=dataset.collate)
        self.output_file_name = args.out_emd_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = args.dim
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.label, self.n_label = g_hin.load_label()
        self.initial_lr = args.alpha

        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()

        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):
        print("Training")
        for epoch in range(self.epochs):
            optimizer = optim.SparseAdam(list(self.skip_gram_model.parameters()), lr=self.initial_lr)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))
            running_loss = 0.0
            epoch_loss = 0.0

            n = 0
            for i, sample_batched in enumerate(self.dataloader):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)

                    pos_v = sample_batched[1].to(self.device)

                    neg_v = sample_batched[2].to(self.device)

                    scheduler.step()

                    optimizer.zero_grad()

                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)

                    loss.backward()

                    optimizer.step()
                    # running_loss = running_loss * 0.9 + loss.item() * 0.1
                    epoch_loss += loss.item()
                    # if i > 0 and i % 50 == 0:

                    #     print(" Loss: " + str(running_loss))
                    n = i
            print("epoch:" + str(epoch) + " Loss: " + str(epoch_loss / n))
            self.skip_gram_model.eva_embedding(self.data.word2id, self.label, self.n_label)
            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)


