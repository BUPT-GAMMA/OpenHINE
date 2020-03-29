# OpenHINE

This is an open-source toolkit for Heterogeneous Information Network Embedding(OpenHINE) with version 0.1. We can train and test the model more easily. It provides implementations of many popular models, including: DHNE, HAN, HeGAN, HERec, HIN2vec, Metapath2vec, MetaGraph2vec, RHINE. More materials can be found in [www.shichuan.org](http://www.shichuan.org).

**convenience provided:**

- ​	easy to train and evaluate
- ​	able to extend new/your datasets and models
- ​	the latest model available: HAN、HeGAN and so on	

#### Contributors：

DMGroup from BUPT: Tianyu Zhao, Meiqi Zhu, Jiawei Liu, Nian Liu, Guanyi Chu, Jiayue Liu, Xiao Wang, Cheng Yang, Linmei Hu, Chuan Shi.

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

## Model

#### Available

##### 	DHNE

​		Structural Deep Embedding for Hyper-Networks 

​		[DHNE AAAI 2018]

​		src code:https://github.com/tadpole/DHNE

##### 	HAN

​		Heterogeneous Graph Attention Network

​		 [HAN WWW 2019]

​		src code:https://github.com/Jhy1993/HAN

##### HeGAN

​		Adversarial Learning on Heterogeneous Information Network 

​		[HeGAN KDD 2019]

​		src code:https://github.com/librahu/HeGAN

##### 	HERec

​		Heterogeneous Information Network Embedding for Recommendation 

​		[HERec TKDE 2018]

​		src code:https://github.com/librahu/HERec

###### 		*spec para: 

​			metapath_list: pap|psp	(split by "|")

##### 	HIN2vec

​		HIN2Vec: Explore Meta-paths in Heterogeneous Information Networks for Representation Learning 

​		[HIN2Vec CIKM 2017]

​		src code:https://github.com/csiesheep/hin2vec

##### 	Metapath2vec	

​		metapath2vec: Scalable Representation Learning for Heterogeneous Networks 

​		[metapath2vec KDD 2017]

​		src code:https://ericdongyx.github.io/metapath2vec/m2v.html

​		the python version implemented by DGL:https://github.com/dmlc/dgl/tree/master/examples/pytorch/metapath2vec 

##### 	MetaGrapth2vec

​		MetaGraph2Vec: Complex Semantic Path Augmented Heterogeneous Network Embedding

​		[MetaGraph2Vec PAKDD 2018]

​		src code:https://github.com/daokunzhang/MetaGraph2Vec

##### 	RHINE

​		Relation Structure-Aware Heterogeneous Information Network Embedding 

​		[RHINE AAAI 2019]

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
| RHINE               | 0.9360   | 0.9316   | 0.7356 |

HAN uses the dataset without features.

## Future work

Note that OpenHINE is just version 0.1 and **still actively under development**, so feedback and contributions are welcome. Feel free to submit your questions as a issue.

In the future, we will contain more models and tasks. We use the assorted deep learning framework, so we want to unify the model with PyTorch. If you have a demo of the above model with PyTorch or want your method added into our toolkit, contract us please.
