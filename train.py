import argparse
from src.config import Config
from src.utils.data_process import *
from src.utils.sampler import *
from src.model.DHNE import *
from src.model.HAN import *
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
    elif args.model == "HAN":
        data_process = HAN_process(g_hin, config.mp_list, args.dataset, config.featype)
        config.out_emd_file += args.dataset + '_node.txt'
        m = HAN(config, data_process)
        m.train()
    else:
        pass
def init_para():
    parser = argparse.ArgumentParser(description="OPEN-HINE")
    parser.add_argument('-d', '--dataset', default='acm', type=str, help="Dataset")
    parser.add_argument('-m', '--model', default='HAN', type=str, help='Train model')
    # parser.add_argument('-t', '--task', default='node_classification', type=str, help='Evaluation task')
    # parser.add_argument('-p', '--metapath', default='pap', type=str, help='Metapath sampling')
    # parser.add_argument('-s', '--save', default='1', type=str, help='save temproal')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
