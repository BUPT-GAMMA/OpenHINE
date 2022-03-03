# OpenHINE

This is an open-source toolkit for Heterogeneous Information Network Embedding(OpenHINE) with version 0.1. We can train and test the model more easily. It provides implementations of many popular models, including: DHNE, HAN, HeGAN, HERec, HIN2vec, Metapath2vec, MetaGraph2vec, RHINE. More materials can be found in [www.shichuan.org](http://www.shichuan.org).

We build a new toolkit [OpenHGNN](https://github.com/BUPT-GAMMA/OpenHGNN), which is a high-level package built on top of DGL. It will have Better Extensibility, Better Encapsulation and More Effiencient. And it includes two embedding models, Metapath2vec and HeRec.

**convenience provided:**

- ​	easy to train and evaluate
- ​	able to extend new/your datasets and models
- ​	the latest model available: HAN、HeGAN and so on	

#### Contributors：

DMGroup from BUPT: Tianyu Zhao, Meiqi Zhu, Nian Liu, Jiawei Liu, Hongrui Liu, Guanyi Chu, Jiayue Liu, Jianan Zhao, Xiao Wang, Cheng Yang, Chuan Shi.

## Get started

### Requirements and Installation

- Python version >= 3.6


- PyTorch version >= 1.4.0

- TensorFlow version  >= 1.14

- Keras version >= 2.3.1


### config/Usage

##### Input parameter

```python
python train.py -m model_name -d dataset_name
```

e.g.

```python
python train.py -m Metapath2vec -d acm
```



##### Model Setup

The model parameter could be modified in the file ( ./src/config.ini ).

- ###### 	common parameter


​	--alpha:	learning rate

​	--dim:	dimension of output

​	--epoch: the number of iterations	

​	--num_workers:number of workers for dataset loading (It should be set to 0, if you are in trouble with Windows OS.)

​	etc...

- ###### 	specific parameter


​	--metapath:	the metapath selected

​	--neg_num:	the number of negative samples		

​	etc...

### Datasets

If you want to train your own dataset, create the file (./dataset/your_dataset_name/edge.txt) and the format   is as follows：

###### 	input:	edge

​		src_node_id	dst_node_id	edge_type	weight

​	e.g.

		19	7	p-c	2
		19	7	p-a	1
		11	0	p-c	1
		0	11	c-p	1

PS：The input graph is directed and the undirected needs to be transformed into directed graph.

###### Input:	feature	

​		number_of_nodes embedding_dim

​		node_name dim1 dim2 

e.g.

```
11246	2
a1814 0.06386946886777878 -0.04781734198331833
a0 ... ...
```



## Model

#### Available

##### 	[DHNE AAAI 2018]

​		Structural Deep Embedding for Hyper-Networks 

​		src code:https://github.com/tadpole/DHNE

##### 	[HAN WWW 2019]

​		Heterogeneous Graph Attention Network

​		Add feature.txt into the input folder or set the parameter "featype": "adj", which means that you will use adjacency matrix as your feature.

​		src code:https://github.com/Jhy1993/HAN

##### [HeGAN KDD 2019]

​		Adversarial Learning on Heterogeneous Information Network 

​		src code:https://github.com/librahu/HeGAN

##### 	[HERec TKDE 2018]

​		Heterogeneous Information Network Embedding for Recommendation 

​		src code:https://github.com/librahu/HERec

###### 		*spec para: 

​			metapath_list: pap|psp	(split by "|")

##### 	[HIN2Vec CIKM 2017]

​		HIN2Vec: Explore Meta-paths in Heterogeneous Information Networks for Representation Learning 

​		src code:https://github.com/csiesheep/hin2vec

##### 	[metapath2vec KDD 2017]

​		metapath2vec: Scalable Representation Learning for Heterogeneous Networks 

​		src code:https://ericdongyx.github.io/metapath2vec/m2v.html

​		the python version implemented by DGL:https://github.com/dmlc/dgl/tree/master/examples/pytorch/metapath2vec 

##### 	[MetaGraph2Vec PAKDD 2018]

​		MetaGraph2Vec: Complex Semantic Path Augmented Heterogeneous Network Embedding

​		src code:https://github.com/daokunzhang/MetaGraph2Vec

##### [PTE KDD 2015]

​		PTE: Predictive Text Embedding through Large-scale Heterogeneous Text Networks

​		src code:https://github.com/mnqu/PTE

##### 	[RHINE AAAI 2019]

​		Relation Structure-Aware Heterogeneous Information Network Embedding 

​		only supported in the Linux

​		src code:https://github.com/rootlu/RHINE 

## Output

#### Test

```python
python test.py -d dataset_name -m model_name -n file_name
```

The output embedding file name can be found in (./output/embedding/model_name/) .

e.g.

```python
python test.py -d dblp -m HAN -n node.txt
```

###### output:	embedding	

​		number_of_nodes embedding_dim

​		node_name dim1 dim2 

e.g.

```
11246	2
a1814 0.06386946886777878 -0.04781734198331833
a0 ... ...
```



## Evaluation/Task

| ACM dataset       | Micro-F1 | Macro-F1 | NMI    |
| ----------------- | -------- | -------- | ------ |
| DHNE              | 0.7201   | 0.7007   | 0.3280 |
| HAN               | 0.8401   | 0.8362   | 0.4241 |
| HeGAN             | 0.8308   | 0.8276   | 0.4335 |
| HERec             | 0.8308   | 0.8304   | 0.3618 |
| HIN2vec           | 0.8458   | 0.8449   | 0.4148 |
| Metapath2vec(PAP) | 0.7823   | 0.7725   | 0.2828 |
| MetaGraph2vec     | 0.8085   | 0.8019   | 0.5095 |
| PTE               | 0.7624   | 0.7543   | 0.3781 |
| RHINE             | 0.7699   | 0.7571   | 0.3970 |

| DBLP dataset        | Micro-F1 | Macro-F1 | NMI    |
| ------------------- | -------- | -------- | ------ |
| DHNE                | ---      | ---      | ---    |
| HAN                 | 0.8325   | 0.8141   | 0.3415 |
| HeGAN               | 0.9414   | 0.9364   | 0.7898 |
| HERec               | 0.9249   | 0.9214   | 0.3412 |
| HIN2vec             | 0.9495   | 0.9460   | 0.3924 |
| Metapath2vec(APCPA) | 0.9483   | 0.9448   | 0.7786 |
| MetaGraph2vec       | 0.9138   | 0.9093   | 0.6136 |
| PTE                 | 0.9335   | 0.9301   | 0.3280 |
| RHINE               | 0.9360   | 0.9316   | 0.7356 |

HAN uses the dataset without features.

## Future work
We will use the [dgl](https://github.com/dmlc/dgl) as our backend. And the OpenHINE will not be updated. We will be dedicated in building the new toolkit OpenHGNN, which is a high-level package built on top of DGL. It will have Better Extensibility, Better Encapsulation and More Effiencient.
