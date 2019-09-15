import numpy as np
import tensorflow as tf
import math
import time
import threading
import functools
import struct


class Testcase(object):
    def __init__(self):
        self.h = 0
        self.r = 0
        self.w = 0
        self.t = 0
        self.candidates = []


class PME(object):
    def __init__(self, input_edge, node2id_dict, relation2id_dict, dimension, dimensionR, loadBinaryFlag,
                 outBinaryFlag, threads, nbatches, train_times, no_validate, alpha, margin, output_embfold):
        self.entityTotal = len(node2id_dict)
        self.relationTotal = len(relation2id_dict)
        self.tripleTotal = 0
        self.entity2id = node2id_dict
        self.relation2id = relation2id_dict
        self.id2entity = {}
        self.id2relation = {}
        self.dimension = dimension
        self.dimensionR = dimensionR
        self.entityVec = np.zeros((self.entityTotal, self.dimension))
        self.entityVecDao = np.zeros((self.entityTotal, self.dimension))
        self.relationVec = np.zeros((self.relationTotal, self.dimensionR))
        self.relationVecDao = np.zeros((self.relationTotal, self.dimensionR))
        self.matrix = np.zeros((self.relationTotal, self.dimensionR, self.dimension))
        self.matrixDao = np.zeros((self.relationTotal, self.dimensionR, self.dimension))
        self.freqRel = np.zeros(self.relationTotal)
        self.freqEnt = np.zeros(self.entityTotal)
        # h, t, r, w
        self.trainHead = np.array([0, 0, 0, 0])
        self.trainTail = np.array([0, 0, 0, 0])
        self.trainList = np.array([0, 0, 0, 0])
        self.lefHead = np.zeros(self.entityTotal)
        self.rigHead = np.zeros(self.entityTotal)
        self.lefTail = np.zeros(self.entityTotal)
        self.rigTail = np.zeros(self.entityTotal)
        self.left_mean = np.zeros(self.relationTotal)
        self.right_mean = np.zeros(self.relationTotal)
        self.initPath = ""
        self.loadPath = ""
        self.loadBinaryFlag = loadBinaryFlag
        self.outBinaryFlag = outBinaryFlag
        self.note = ""
        self.threads = threads
        self.next_random = 0
        self.nbatches = nbatches
        self.transRBatch = 0
        self.trainTimes = train_times
        self.sampling_individual_loss = 0
        self.no_validate = no_validate
        self.validate = [Testcase() for i in range(no_validate)]
        self.res = 0
        self.res_u2u = 0
        self.res_u2b = 0
        self.res_b2a = 0
        self.res_b2c = 0
        self.last_res = 0
        self.last_res_u2u = 0
        self.last_res_u2b = 0
        self.last_res_b2a = 0
        self.last_res_b2c = 0
        self.no_sampledRel_all = 0
        self.no_sampledRel_u2u = 0
        self.no_sampledRel_u2b = 0
        self.no_sampledRel_b2a = 0
        self.no_sampledRel_b2c = 0
        self.tmpValue = 0
        self.alpha = alpha
        self.margin = margin
        self.outPath = output_embfold
        self.note = ""

        for i in self.entity2id:
            self.id2entity[self.entity2id[i]] = i
        for i in self.relation2id:
            self.id2relation[self.relation2id[i]] = i

        for i in range(self.entityTotal):
            for j in range(self.dimension):
                self.entityVec[i][j] = randn(0, 1.0/self.dimension, -6/math.sqrt(self.dimension), 6/math.sqrt(self.dimension))
            self.entityVec[i] = norm(self.entityVec[i])
        for i in range(self.relationTotal):
            for j in range(self.dimensionR):
                self.relationVec[i][j] = randn(0, 1.0/self.dimensionR, -6/math.sqrt(self.dimensionR), 6/math.sqrt(self.dimensionR))
        for i in range(self.relationTotal):
            for j in range(self.dimensionR):
                for k in range(self.dimension):
                    self.matrix[i][j][k] = randn(0, 1.0/self.dimension, -6/math.sqrt(self.dimension), 6/math.sqrt(self.dimension))

        with open(input_edge) as file:
            file = file.readlines()
            for line in file:
                self.tripleTotal = self.tripleTotal + 1
                token = line.strip('\n').split("\t")
                tmplist = [int(token[0]), int(token[1]), self.relation2id[token[2][0] + token[2][2]], int(token[3])]
                self.trainList = np.row_stack((self.trainList, tmplist))
                self.trainHead = np.row_stack((self.trainHead, tmplist))
                self.trainTail = np.row_stack((self.trainTail, tmplist))
                self.freqEnt[tmplist[0]] += 1
                self.freqEnt[tmplist[1]] += 1
                self.freqRel[tmplist[2]] += 1

            self.trainList = np.delete(self.trainList, 0, axis=0)
            self.trainHead = np.delete(self.trainHead, 0, axis=0)
            self.trainTail = np.delete(self.trainTail, 0, axis=0)

            self.trainHead = sorted(self.trainHead, key=functools.cmp_to_key(cmp_head))
            self.trainTail = sorted(self.trainTail, key=functools.cmp_to_key(cmp_tail))
            self.trainList = sorted(self.trainList, key=functools.cmp_to_key(cmp_list))

        for i in range(1, self.tripleTotal):
            if self.trainTail[i][1] != self.trainTail[i-1][1]:
                self.rigTail[self.trainTail[i-1][1]] = i-1
                self.lefTail[self.trainTail[i][1]] = i
            if self.trainHead[i][0] != self.trainHead[i-1][0]:
                self.rigHead[self.trainHead[i-1][0]] = i-1
                self.lefHead[self.trainHead[i][0]] = i
        self.rigHead[self.trainHead[self.tripleTotal - 1][0]] = self.tripleTotal - 1
        self.rigTail[self.trainTail[self.tripleTotal - 1][1]] = self.tripleTotal - 1

        for i in range(self.entityTotal):
            for j in range(int(self.lefHead[i]+1), int(self.rigHead[i]+1)):
                if self.trainHead[j][2] != self.trainHead[j-1][2]:
                    self.left_mean[self.trainHead[j][2]] += 1.0
            if self.lefHead[i] <= self.rigHead[i]:
                self.left_mean[self.trainHead[int(self.lefHead[i])][2]] += 1.0
            for j in range(int(self.lefTail[i]+1), int(self.rigTail[i]+1)):
                if self.trainTail[j][2] != self.trainTail[j-1][2]:
                    self.right_mean[self.trainTail[j][2]] += 1.0
            if self.lefTail[i] <= self.rigTail[i]:
                self.right_mean[self.trainTail[int(self.lefTail[i])][2]] += 1.0

        for i in range(self.relationTotal):
            self.left_mean[i] = self.freqRel[i]/self.left_mean[i]
            self.right_mean[i] = self.freqRel[i]/self.right_mean[i]

        # if self.initPath != "":
        #     for i in range(self.relationTotal):
        #         for j in range(self.dimensionR):
        #             for k in range(self.dimension):
        #                 if j == k:
        #                     self.matrix[i][j][k] = 1
        #                 else:
        #                     self.matrix[i][j][k] = 0
        print("Init finished!")

    def load(self):
        if self.loadBinaryFlag:
            self.load_binary()
            return
        with open(self.loadPath+"entity2vec"+self.note+".vec") as file:
            i = 0
            file = file.readlines()
            for line in file:
                self.entityVec[i] = line.strip('\n').split('\t')
                i = i + 1
        with open(self.loadPath+"relation2vec"+self.note+".vec") as file:
            i = 0
            file = file.readlines()
            for line in file:
                self.relationVec[i] = line.strip('\n').split('\t')
                i = i + 1
        # with open(self.loadPath+"A"+self.note+".vec") as file:
        #     i = 0
        #     file = file.readlines()
        #     for line in file:
        #         self.matrix[i][:][j] = line.strip('\n').split('\t')
        #         i = i + 1

    def train(self):
        transRLen = self.tripleTotal
        self.transRBatch = transRLen / self.nbatches
        self.next_random = np.zeros(self.threads)
        self.tmpValue = np.zeros(self.threads * self.dimensionR * 4)
        self.relationVecDao = np.copy(self.relationVec)
        self.entityVecDao = np.copy(self.entityVec)
        self.matrixDao = np.copy(self.matrix)

        t1 = time.time()
        for epoch in range(self.trainTimes):
            if epoch == 10:
                diff = time.time() - t1
                print("Wall Time = " + diff)
            if epoch != 0 and epoch % 200 == 0:
                self.validation_topk(20)
            self.res = self.res_u2u = self.res_u2b = self.res_b2a = self.res_b2c = 0
            self.no_sampledRel_all = self.no_sampledRel_u2u = \
                self.no_sampledRel_u2b = self.no_sampledRel_b2a = self.no_sampledRel_b2c = 0
            for batch in range(self.nbatches):
                pt = np.zeros(self.threads)
                for a in range(self.threads):
                    pt[a] = threading.Thread(target=self.trainMode, args=(self, a))
                    pt[a].start()
                for a in range(self.threads):
                    pt[a].join()
                self.relationVec = self.relationVecDao
                self.entityVec = self.entityVecDao
                self.matrix = self.matrixDao
            self.last_res = self.res
            self.last_res_u2u = self.res_u2u
            self.last_res_u2b = self.res_u2b
            self.last_res_b2a = self.res.b2a
            self.last_res_b2c = self.res.b2c

            print("epoch: ", epoch, "overall loss: ", self.res, "u2u: ", self.res_u2u,
                  "u2b: ", self.res_u2b, "b2a: ", self.res_b2a, "b2c: ", self.res_b2c)
            print("sampled edges: ", "u2u: ", self.no_sampledRel_u2u, "u2b: ", self.no_sampledRel_u2b, "b2a: ",
                  self.no_sampledRel_b2a, "b2c: ", self.no_sampledRel_b2c)
            if epoch == self.trainTimes - 1:
                self.out()
                self.validation_topk(20)
                self.validation_topk(15)
                self.validation_topk(10)
                self.validation_topk(5)
                self.validation_topk(1)
        print("Train finished!")

    def trainMode(self, con):
        idd = con
        self.next_random = np.random.rand()
        tmp = self.tmpValue[idd * self.dimensionR * 4:]
        for k in range(self.transRBatch/self.threads, -1, -1):
            is_any_loss_zero = self.last_res_u2u == 0 or self.last_res_u2b == 0 or \
                               self.last_res_b2a == 0 or self.last_res_b2c == 0
            if self.sampling_individual_loss and not is_any_loss_zero:
                prob_u2u = 1000 * self.last_res_u2u / self.last_res
                prob_u2b = 1000 * self.last_res_u2b / self.last_res
                prob_b2a = 1000 * self.last_res_b2a / self.last_res
                prob_b2c = 1000 * self.last_res_b2c / self.last_res
            else:
                prob_u2u = 1000 * 303722/560874
                prob_u2b = 1000 * 192400/560874
                prob_b2a = 1000 * 32273/560874
                prob_b2c = 1000 * 32479/560874
            pr = self.randd(idd) % 1000
            if pr < prob_u2u:
                next_sample = 0
            elif pr < (prob_u2u + prob_u2b):
                next_sample = 1
            elif pr < (prob_u2u + prob_u2b + prob_b2a):
                next_sample = 2
            else:
                next_sample = 3

            while 1:
                i = self.rand_max(idd, self.tripleTotal)
                sampledRel = self.trainList[2]
                if sampledRel == next_sample:
                    break

            self.no_sampledRel_all += 1
            if sampledRel == 0:
                self.no_sampledRel_u2u += 1
            if sampledRel == 1:
                self.no_sampledRel_u2b += 1
            if sampledRel == 2:
                self.no_sampledRel_b2a += 1
            if sampledRel == 3:
                self.no_sampledRel_b2c += 1

            for m in range(self.M):
                j = self.corrupt_head(idd, self.trainList[i][0], self.trainList[i][1])
                self.train_kb(self.trainList[i][0], self.trainList[i][1], self.trainList[i][2],
                              self.trainList[i][3], self.trainList[i][0], j, self.trainList[i][2], tmp)

    def validation_topk(self, topk):
        i = 0
        hit_u2u = 0
        hit_u2b = 0
        hit_b2a = 0
        hit_b2c = 0
        while i < self.no_validate:
            h = self.validate[i].h
            t = self.validate[i].t
            r = self.validate[i].r
            arr = []
            for j in range(5000):
                arr.append(self.validate[i].candidates[j])
            arr.append(t)
            distances = []
            results = [i for i in range(len(arr))]
            for i in range(len(arr)):
                distances.append(self.calc_distance(h, r, arr[i]))

            results.sort(lambda x: distances(x))
            results = results[:topk]
            if 5000 in results:
                if r == 0:
                    hit_u2u += 1
                if r == 1:
                    hit_u2b += 1
                if r == 2:
                    hit_b2a += 1
                if r == 3:
                    hit_b2c += 1
            i += 1
        print("top@", topk, ": u2u: ", hit_u2u*1.0/(self.no_validate/4), "\tu2b: ", hit_u2b*1.0/(self.no_validate/4),
              "\tb2a: ", hit_b2a*1.0/(self.no_validate/4), "\tb2c: ", hit_b2c*1.0/(self.no_validate/4))

    def calc_distance(self, h, r, t):
        tmp = np.zeros(self.dimensionR)
        tmp1 = tmp
        tmp2 = np.zeros(self.dimensionR)
        distance = 0

        Relation_Matrix = r * self.dimension * self.dimensionR
        h_vec = h * self.dimension
        t_vec = t * self.dimension

        for i in range(self.dimensionR):
            tmp1[i] = tmp2[i] = 0
            for j in range(self.dimension):
                tmp1[i] += self.matrix[Relation_Matrix + j] * self.entityVec[h_vec + j]
                tmp2[i] += self.matrix[Relation_Matrix + j] * self.entityVec[t_vec + j]
            Relation_Matrix += self.dimension
            distance += math.pow((tmp1[i] - tmp2[i]),2)
        return distance

    def randd(self, idd):
        self.next_random[idd] = self.next_random[idd] * 25214903917 + 11
        return self.next_random[idd]

    def rand_max(self, idd, x):
        res = self.randd(idd) % x
        while res < 0:
            res += x
        return res

    def calc_sum(self, e1, e2, rel, tmp1, tmp2):
        lastM = rel * self.dimension * self.dimensionR
        last1 = e1 * self.dimension
        last2 = e2 * self.dimension
        sumx = 0
        for ii in range(self.dimensionR):
            tmp1[ii] = tmp2[ii] = 0
            for jj in range(self.dimension):
                tmp1[jj] += self.matrix[lastM + jj] * self.entityVec[last1 + jj]
                tmp2[jj] += self.matrix[lastM + jj] * self.entityVec[last2 + jj]
            lastM += self.dimension
            sumx += math.pow((tmp1[ii] - tmp2[ii]), 2)
        return sumx

    def gradient(self, e1_a, e2_a, rel_a, belta, same, tmp1, tmp2):
        lasta1 = e1_a * self.dimension
        lasta2 = e2_a * self.dimension
        lastar = rel_a * self.dimensionR
        lastM = rel_a * self.dimensionR * self.dimension
        for ii in range(self.dimensionR):
            x = 2 * (tmp2[ii] - tmp1[ii])
            x = belta * self.alpha * x
            for jj in range(self.dimension):
                self.matrixDao[lastM + jj] -= x * (self.entityVec[lasta1 + jj] - self.entityVec[lasta2 + jj])
                self.entityVecDao[lasta1 + jj] -= x * self.matrix[lastM + jj]
                self.entityVecDao[lasta2 + jj] += x * self.matrix[lastM + jj]
            lastM += self.dimension

    def train_kb(self, e1_a, e2_a, rel_a, weight, e1_b, e2_b, rel_b, tmp):
        sum1 = self.calc_sum(e1_a, e2_a, rel_a, tmp, tmp[self.dimensionR:])
        sum2 = self.calc_sum(e1_b, e2_b, rel_b, tmp[self.dimensionR*2:], tmp[self.dimensionR*3:])
        loss_per_pair = (self.margin + sum1 - sum2) if (self.margin + sum1 - sum2) > 0 else 0

        if loss_per_pair > 0:
            self.res += loss_per_pair
            if rel_a != rel_b:
                print("relation is not same when training...!")
            if rel_a == 0:
                self.res_u2u += loss_per_pair
            elif rel_a == 1:
                self.res_u2b += loss_per_pair
            elif rel_a == 2:
                self.res_b2a += loss_per_pair
            elif rel_a == 3:
                self.res_b2c += loss_per_pair
            self.gradient(e1_a, e2_a, rel_a, -1, 1, tmp, tmp[self.dimensionR:])
            self.gradient(e1_b, e2_b, rel_b, 1, 1, tmp[self.dimensionR*2:], tmp[self.dimensionR*3:])

    def corrupt_head(self, idd, h, r):
        lef = self.lefHead[h] - 1
        rig = self.rigHead[h]
        while lef+1 < rig:
            mid = (lef + rig) >> 1
            if self.trainHead[mid][2] >= r:
                rig = mid
            else:
                lef = mid
        ll = rig

        lef = self.lefHead[h]
        rig = self.rigHead[h] + 1
        while lef+1 < rig:
            mid = (lef + rig) >> 1
            if self.trainHead[mid][2] <= r:
                lef = mid
            else:
                rig = mid
        rr = lef

        tmp = self.rand_max(idd, self.entityTotal - (rr - ll + 1))
        if tmp < self.trainHead[ll][1]:
            return tmp
        if tmp < self.trainHead[rr][1] - rr + ll - 1:
            return tmp + rr - ll + 1
        lef = ll
        rig = rr + 1
        while lef + 1 < rig:
            mid = (lef + rig) >> 1
            if self.trainHead[mid][1] - mid + ll - 1 < tmp:
                lef = mid
            else:
                rig = mid
        return tmp + lef - ll + 1

    def load(self):
        if self.loadBinaryFlag:
            self.load_binary()
            return

        with open(self.loadPath + "relation2vec" + self.note + ".vec", mode='r') as fin:
            for i in range(self.relationTotal):
                for ii in range(self.dimensionR):
                    tmp = fin.readline()
                    self.relationVec[i] = tmp.strip('\n').split("\t")
        with open(self.outPath + "entity2vec" + self.note + ".vec", mode='r') as fin:
            for i in range(self.entityTotal):
                for ii in range(self.dimension):
                    tmp = fin.readline()
                    self.entityVec[i] = tmp.strip('\n').split("\t")
        with open(self.outPath + "A" + self.note + ".vec", mode='r') as fin:
            for i in range(self.relationTotal):
                for j in range(self.dimensionR):
                    for k in range(self.dimension):
                        tmp = fin.readline()
                        self.M[i][j] = tmp.strip('\n').split("\t")

    def load_binary(self):
        with open(self.outPath + "relation2vec" + self.note + ".bin", mode="rb") as f2:
            len = self.relationTotal * self.dimension
            tot = 0
            while tot < len:
                self.relationVec[tot//4] = struct.unpack("f", f2.read(4))[0]
                tot += 4
        with open(self.outPath + "entity2vec" + self.note + ".bin", mode="rb") as f3:
            len = self.entityTotal * self.dimension
            tot = 0
            while tot < len:
                self.entityVec[tot//4] = struct.unpack("f", f3.read(4))[0]
                tot += 4
        with open(self.outPath + "A" + self.note + ".bin", mode="rb") as f1:
            len = self.relationTotal * self.dimension * self.dimensionR
            tot = 0
            while tot < len:
                self.matrix[tot//4] = struct.unpack("f", f1.read(4))[0]
                tot += 4

    def out(self):
        if self.outBinaryFlag:
            self.out_binary()
            return

        with open(self.outPath + "relation2vec" + self.note + ".vec", mode='w') as fin:
            for i in range(self.relationTotal):
                for ii in range(self.dimensionR):
                    fin.write(str(self.relationVec[i][ii]))
                    fin.write('\t')
                fin.write('\n')
        with open(self.outPath + "entity2vec" + self.note + ".vec", mode='w') as fin:
            for i in range(self.entityTotal):
                for ii in range(self.dimension):
                    fin.write(str(self.entityVec[i][ii]))
                    fin.write('\t')
                fin.write('\n')
        with open(self.outPath + "A" + self.note + ".vec", mode='w') as fin:
            for i in range(self.relationTotal):
                for j in range(self.dimensionR):
                    for k in range(self.dimension):
                        fin.write(str(self.matrix[i][j][k]))
                        fin.write('\t')
                    fin.write('\n')

    def out_binary(self):
        with open(self.outPath + "relation2vec" + self.note + ".bin", mode="wb") as f2:
            content = self.relationVec.to_bytes(4, 'big')
            f2.write(content)
        with open(self.outPath + "entity2vec" + self.note + ".bin", mode="wb") as f3:
            content = self.entityVec.to_bytes(4, 'big')
            f3.write(content)
        with open(self.outPath + "A" + self.note + ".bin", mode="wb") as f1:
            content = self.matrix.to_bytes(4, 'big')
            f1.write(content)


def normal(x, miu, sigma):
    return 1.0/math.sqrt(2*math.pi)/sigma*math.exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma))


def randn(miu, sigma, minx, maxx):
    while 1:
        x = np.random.uniform(minx, maxx)
        y = normal(x, miu, sigma)
        dScope = np.random.uniform(0, normal(miu, miu, sigma))
        if dScope > y:
            break
    return x


def norm(x):
    return x/np.sqrt(np.sum(x**2))


def cmp_head(a, b):
    if a[0] < b[0]:
        return 1
    elif a[0] == b[0] and a[2] < b[2]:
        return 1
    elif a[0] == b[0] and a[2] == b[2] and a[1] < b[1]:
        return 1
    else:
        return -1


def cmp_tail(a, b):
    if a[1] < b[1]:
        return 1
    elif a[1] == b[1] and a[2] < b[2]:
        return 1
    elif a[1] == b[1] and a[2] == b[2] and a[0] < b[0]:
        return 1
    else:
        return -1


def cmp_list(a, b):
    if max(a[0], a[1]) < max(b[0], b[1]):
        return 1
    else:
        return -1
