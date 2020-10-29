import argparse
from src.config import Config
from src.utils.data_process import *
from src.model.RHINE import *
from src.model.Metapath2vec import *
from src.utils.sampler import *
from src.utils.hete_random_walk import *
from src.utils.utils import *
from src.model.DHNE import *
# from src.model import HHNE
# from src.model.MetaGraph2vec import *
#from src.model.PME import *
from src.model.HERec import DW
from src.model.HIN2vec import *
from src.model.HAN import *
from src.model.HeGAN import HeGAN
from src.model.PTE import *
import warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings('ignore')

seed = 0
def main():
    args = init_para()
    config_file = ["./src/config.ini"]
    config = Config(config_file, args)

    g_hin = HIN(config.input_fold, config.data_type, config.relation_list)

    # Model selection
    if args.model == "RHINE":
        g_hin.load_matrix()
        g_hin.generate_matrix(config.combination)
        RHINEdp = RHINEDataProcess(config, g_hin)
        RHINEdp.generate_triples()
        RHINEdp.merge_triples(config.relation_category)
        print("Train")
        TrainRHINE(config, g_hin.node2id_dict)
    elif args.model == "Metapath2vec":
        config.temp_file += args.dataset + '_' + config.metapath + '.txt'
        config.out_emd_file += args.dataset + '_' + config.metapath + '.txt'

        random_walk_based_mp(g_hin, config.metapath, config.num_walks, config.walk_length, config.temp_file)
        m2v = Metapath2VecTrainer(config, g_hin)
        m2v.train()

    elif args.model == "HeteSpaceyWalk":
        config.temp_file += args.dataset + '_' + config.metapath + '.txt'
        config.out_emd_file += args.dataset + '_' + config.metapath + '.txt'
        random_walk_spacey_mp(g_hin, config.metapath, config.data_type,
                              config.num_walks, config.walk_length, config.temp_file, config.beta)
        m2v = Metapath2VecTrainer(config)
        m2v.train()
    elif args.model == "DHNE":
        hyper_edge_sample(g_hin, output_datafold=config.temp_file, scale=config.scale, tup=config.triple_hyper)
        dataset = read_data_sets(train_dir=config.temp_file)
        dim_feature = [sum(dataset.train.nums_type) - n for n in dataset.train.nums_type]
        Process(dataset, dim_feature, embedding_size=config.dim, hidden_size=config.hidden_size,
                learning_rate=config.alpha, alpha=config.alpha, batch_size=config.batch_size,
                num_neg_samples=config.neg_num, epochs_to_train=config.epochs, output_embfold=config.out_emd_file,
                output_modelfold=config.output_modelfold, prefix_path=config.prefix_path, reflect=g_hin.matrix2id_dict)
    # elif args.model == "HHNE":
    #     random_walk_txt = config.temp_file + args.dataset + '-' + config.metapath + '.txt'
    #     node_type_mapping_txt = config.temp_file + 'node_type_mapping.txt'
    #     config.out_emd_file += args.dataset + '-' + config.metapath + '.txt'
    #     print("Metapath walking!")
    #     if len(config.metapath) == 3:
    #         # data = random_walk_three(config.num_walks, config.walk_length, config.metapath, g_hin, random_walk_txt)
    #         data = random_walk_three(1, 5, config.metapath, g_hin, random_walk_txt)
    #     elif len(config.metapath) == 5:
    #         data = random_walk_five(config.num_walks, config.walk_length, config.metapath, g_hin, random_walk_txt)
    #
    #     node_type_mapping_txt = g_hin.node_type_mapping(node_type_mapping_txt)
    #     dataset = HHNE.Dataset(random_walk_txt=random_walk_txt,window_size=config.window_size)
    #     print("Train" + str(len(dataset.index2nodeid)))
    #     pos_holder, tar_holder, tag_holder, pro_holder, grad_pos, grad_tar = HHNE.bulid_model(EMBED_SIZE=config.dim)
    #     HHNE.TrainHHNE(pos_holder, tar_holder, tag_holder, pro_holder, grad_pos, grad_tar, dataset,
    #               BATCH_SIZE=config.batch_size, NUM_EPOCHS=config.epochs, NUM_SAMPLED=config.neg_num,
    #               VOCAB_SIZE=len(dataset.nodeid2index), EMBED_SIZE=config.dim, startingAlpha=config.alpha,
    #               lr_decay=config.lr_decay, output_embfold=config.out_emd_file)
    elif args.model == "MetaGraph2vec":
        config.temp_file += 'graph_rw.txt'
        config.out_emd_file += args.dataset + '_node.txt'
        mgg = MetaGraphGenerator()
        if args.dataset == "acm":
            mgg.generate_random_three(config.temp_file, config.num_walks, config.walk_length, g_hin.node,
                                      g_hin.relation_dict)
        elif args.dataset == "dblp":
            mgg.generate_random_four(config.temp_file, config.num_walks, config.walk_length, g_hin.node,
                                     g_hin.relation_dict)
        model = Metapath2VecTrainer(config,g_hin)
        print("Training")
        model.train()
    # elif args.model == "PME":
    #     pme = PME(
    #         g_hin.input_edge,
    #         g_hin.node2id_dict,
    #         g_hin.relation2id_dict,
    #         config.dim,
    #         config.dimensionR,
    #         config.loadBinaryFlag,
    #         config.outBinaryFlag,
    #         config.num_workers,
    #         config.nbatches,
    #         config.epochs,
    #         config.no_validate,
    #         config.alpha,
    #         config.margin,
    #         config.M,
    #         config.out_emd_file
    #     )
    #     # pme.load()
    #     pme.train()
    #     pme.out()
    elif args.model == "PTE":
        config.temp_file += args.dataset + '.txt'
        config.out_emd_file += args.dataset + '_node.txt'
        print('PTE')
        data = PTEDataReader(g_hin, config)
        alias_table = AliasSampling(data)
        pte = PTETrainer(g_hin, config, data, alias_table)
        print('Training')
        pte.train()
    elif args.model == "HERec":
        mp_list = config.metapath_list.split("|")
        for mp in mp_list:
            HERec_gen_neighbour(g_hin, mp, config.temp_file)
            config.input = config.temp_file + mp + ".txt"
            config.out_put = config.out_emd_file + mp + ".txt"
            DW(config)
        HERec_union_metapth(config.out_emd_file, mp_list, len(g_hin.node[mp_list[0][0]]), config.dim)
    elif args.model == "HIN2vec":
        HIN2vec(g_hin, config.out_emd_file, config)
    elif args.model == "HAN":
        data_process = HAN_process(g_hin, config.mp_list, args.dataset, config.featype)
        config.out_emd_file += args.dataset + '_node.txt'
        m = HAN(config, data_process)
        m.train()
    elif args.model == "HeGAN":
        model = HeGAN(g_hin, args, config)
        model.train(config, g_hin.node2id_dict)
    else:
        pass
    # evaluation
    # if args.task == 'node_classification':


def init_para():
    parser = argparse.ArgumentParser(description="OPEN-HINE")
    parser.add_argument('-d', '--dataset', default='acm', type=str, help="Dataset")
    parser.add_argument('-m', '--model', default='MetaGraph2vec', type=str, help='Train model')
    # parser.add_argument('-t', '--task', default='node_classification', type=str, help='Evaluation task')
    # parser.add_argument('-p', '--metapath', default='pap', type=str, help='Metapath sampling')
    # parser.add_argument('-s', '--save', default='1', type=str, help='save temproal')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
