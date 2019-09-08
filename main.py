# from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from src.config import *
from src.data_process import *
from src.model.RHINE import *
from src.model.metapath2vec import *
from src.model.metagraph2vec import *
from src.model.PME import *


def main():

    config_file = ["./src/config.ini"]
    config = Config(config_file)
    # input_edge = 'edge.txt'
    # data_type = apc
    # line_type = p-a+p-c
    # relation_list = p-a+p-c+a-c
    dp = DataProcess(
        config.input_edge,
        config.data_type,
        config.line_type,
        config.relation_list)
    dp.load_node()
    # matrix2id: a ,p and c are each numbered separately.
    # {'a5': 0, 'a8': 1, 'a10': 2, 'a7': 3, 'a9': 4, 'a6': 5, 'p16': 0, 'p13': 1,
    # 'p17': 2, 'p14': 3, 'p12': 4, 'p18': 5, 'p19': 6, 'p15': 7, 'p11': 8, 'c0': 0,
    # 'c2': 1, 'c3': 2, 'c1': 3, 'c4': 4}
    # node2id: a, p and c are numbered together.(a:0-5 p:6-14 c:15-19) (saved to 'node2id.txt')
    # find_dict: inverse mapping of matrix2id (e.g. 'a0':'5')
    # relation2id: (saved to 'relation2id.txt')
    node2id_dict, matrix2id_dict, find_dict, relation2id_dict = dp.renum(
        config.output_datafold)
    # 'p-a': 9*6 'p-c': 9*5
    dp.load_matrix()
    # 'p-a': 9*6 'p-c': 9*5 'a-c': 6*5
    adj_matrix = dp.generate_matrix(config.combination)
    # model train
    if config.model == "RHINE":
        # data reprocessing
        RHINEdp = RHINEDataProcess(
            config.output_datafold,
            find_dict,
            matrix2id_dict,
            adj_matrix,
            config.relation_list)
        RHINEdp.generate_triples()
        RHINEdp.merge_triples(config.RHINE_relation_category)
        print("Train")
        TrainRHINE(config)
    elif config.model == "metapath2vec":
        mpg = MetaPathGenerator()
        mpg.generate_random(
            config.output_randomwalk,
            config.walk_times,
            config.walk_length,
            dp.node,
            find_dict,
            matrix2id_dict,
            adj_matrix,
            config.mp_type)
        dataset = MP2vecDataProcess(
            config.output_randomwalk,
            config.window)
        center_node_placeholder, context_node_placeholder, negative_samples_placeholder, loss = build_model(
            1, len(dataset.nodeid2index), config.dim, config.negative_samples)
        optimizer = traning_op(loss, config.learning_rate)
        train(
            center_node_placeholder,
            context_node_placeholder,
            negative_samples_placeholder,
            loss,
            dataset,
            optimizer,
            NUM_EPOCHS=config.epochs,
            BATCH_SIZE=1,
            NUM_SAMPLED=config.negative_samples,
            care_type=config.care_type,
            LOG_DIRECTORY=config.log_dir,
            LOG_INTERVAL=config.log_interval,
            MAX_KEEP_MODEL=config.max_keep_model)
    elif config.model == "metagraph2vec":
        mpg = MetaGraphGenerator()
        mpg.generate_random(
            config.output_randomwalk,
            config.walk_times,
            config.walk_length,
            dp.node,
            find_dict,
            matrix2id_dict,
            adj_matrix,
            config.mg_type)
        dataset = MG2vecDataProcess(
            config.output_randomwalk,
            config.window)
        center_node_placeholder, context_node_placeholder, negative_samples_placeholder, loss = build_model(
            1, len(dataset.nodeid2index), config.dim, config.negative_samples)
        optimizer = traning_op(loss, config.learning_rate)
        train(
            center_node_placeholder,
            context_node_placeholder,
            negative_samples_placeholder,
            loss,
            dataset,
            optimizer,
            NUM_EPOCHS=config.epochs,
            BATCH_SIZE=1,
            NUM_SAMPLED=config.negative_samples,
            care_type=config.care_type,
            LOG_DIRECTORY=config.log_dir,
            LOG_INTERVAL=config.log_interval,
            MAX_KEEP_MODEL=config.max_keep_model)
    elif config.model == 'PME':
        pass
    else:
        pass


if __name__ == "__main__":
    main()
