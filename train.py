import argparse
from src.config import Config
from src.utils.data_process import *
from src.model.RHINE import *
from src.model.Metapath2vec import *
from src.utils.sampler import *
from src.utils.utils import *
from src.model.DHNE import *
from src.model.HERec import DW
from src.model.HIN2vec import *
from src.model.HAN import *
from src.model.HeGAN import HeGAN
import warnings

warnings.filterwarnings('ignore')


def main():
    args = init_para()
    config_file = ["./src/config.ini"]
    config = Config(config_file, args)

    g_hin = HIN(args.dataset, config.data_type, config.relation_list)

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
        config.temp_file += args.dataset + '-' + config.metapath + '.txt'
        config.out_emd_file += args.dataset + '-' + config.metapath + '.txt'
        print("Metapath walking!")
        if len(config.metapath) == 3:
            data = random_walk_three(config.num_walks, config.walk_length, config.metapath, g_hin, config.temp_file)
        elif len(config.metapath) == 5:
            data = random_walk_five(config.num_walks, config.walk_length, config.metapath, g_hin, config.temp_file)
        m2v = Metapath2VecTrainer(config)
        print("Training")
        m2v.train()
    elif args.model == "DHNE":
        hyper_edge_sample(g_hin, output_datafold=config.temp_file, scale=config.scale, tup=config.triple_hyper)
        dataset = read_data_sets(train_dir=config.temp_file)
        dim_feature = [sum(dataset.train.nums_type) - n for n in dataset.train.nums_type]
        Process(dataset, dim_feature, embedding_size=config.dim, hidden_size=config.hidden_size,
                learning_rate=config.alpha, alpha=config.alpha, batch_size=config.batch_size,
                num_neg_samples=config.neg_num, epochs_to_train=config.epochs, output_embfold=config.out_emd_file,
                output_modelfold=config.output_modelfold, prefix_path=config.prefix_path, reflect=g_hin.matrix2id_dict)

    elif args.model == "MetaGraph2vec":
        config.temp_file += 'graph_rw.txt'
        config.out_emd_file += 'node.txt'
        mgg = MetaGraphGenerator()
        if args.dataset == "acm":
            mgg.generate_random_three(config.temp_file, config.num_walks, config.walk_length, g_hin.node,
                                      g_hin.relation_dict)
        elif args.dataset == "dblp":
            mgg.generate_random_four(config.temp_file, config.num_walks, config.walk_length, g_hin.node,
                                     g_hin.relation_dict)
        model = Metapath2VecTrainer(config)
        print("Training")
        model.train()
    elif args.model == "HERec":
        mp_list = config.metapath_list.split("|")
        for mp in mp_list:
            # HERec_gen_neighbour(g_hin, mp, config.temp_file)
            config.input = config.temp_file + mp + ".txt"
            config.out_put = config.out_emd_file + mp + ".txt"
            DW(config)
        HERec_union_metapth(config.out_emd_file, mp_list, len(g_hin.node[mp_list[0][0]]), config.dim)
    elif args.model == "HIN2vec":
        HIN2vec(g_hin, config.out_emd_file, config)
    elif args.model == "HAN":
        data_process = HAN_process(g_hin, config.mp_list, args.dataset)
        config.out_emd_file += 'node.txt'
        m = HAN(config, data_process)
        m.train()
    elif args.model == "HeGAN":
        model = HeGAN(g_hin, args, config)
        model.train(config, g_hin.node2id_dict)
    else:
        pass



def init_para():
    parser = argparse.ArgumentParser(description="OPEN-HINE")
    parser.add_argument('-d', '--dataset', default='acm', type=str, help="Dataset")
    parser.add_argument('-m', '--model', default='MetaGraph2vec', type=str, help='Train model')


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
