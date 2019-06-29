from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from src.config import *
from src.data_process import *
from src.model.RHINE import *

def main():

    config_file = ["./src/config.ini"]
    config = Config(config_file)
    #处理数据
    dp = DataProcess(config.input_edge, config.data_type, config.line_type, config.relation_list)
    node = dp.load_node()
    node2id_dict, matrix2id_dict,find_dict,relation2id_dict = dp.renum(config.output_datafold)
    dp.load_matrix()
    adj_matrix = dp.generate_matrix(config.combination)
    #模型训练
    if config.model == "RHINE":
        #数据重新处理
        RHINEdp = RHINEDataProcess(config.output_datafold,find_dict, matrix2id_dict , adj_matrix, config.relation_list)
        RHINEdp.generate_triples()
        RHINEdp.merge_triples(config.RHINE_relation_category)
        print("Train")
        TrainRHINE(config)
    else:
        pass

if __name__ == "__main__":
    main()