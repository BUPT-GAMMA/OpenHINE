import configparser
import os


class Config(object):
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        data_path = os.getcwd()
        try:
            conf.read(config_file)
        except BaseException:
            print("failed!")

        # model_choose
        self.model = conf.get("Model", "Model")

        # data_setup
        if conf.has_option("data_setup", "data_type"):
            self.data_type = conf.get("data_setup", "data_type")
        else:
            self.data_type = False

        if conf.has_option("data_setup", "link_type"):
            self.line_type = conf.get("data_setup", "link_type")
        else:
            self.line_type = False

        if conf.has_option("data_setup", "relation_list"):
            self.relation_list = conf.get("data_setup", "relation_list")
        else:
            self.relation_list = False

        if conf.has_option("data_setup", "combination"):
            self.combination = conf.get("data_setup", "combination")
        else:
            self.combination = False

        if conf.has_option("data_setup", "RHINE_relation_category"):
            self.RHINE_relation_category = conf.get(
                "data_setup", "RHINE_relation_category")
        else:
            self.RHINE_relation_category = False

        # training data path
        if conf.has_option("Data_In", "input_edg"):
            self.input_edge = data_path + conf.get("Data_In", "input_edg")
        else:
            self.input_edge = False

        # if conf.has_option("Data_In", "input_id"):
        #     self.input_id = data_path + conf.get("Data_In", "input_id")
        # else:
        #     self.input_id = False

        if conf.has_option("Data_In", "input_fold"):
            self.input_fold = data_path + conf.get("Data_In", "input_fold")
        else:
            self.input_fold = False

        if conf.has_option("Data_Out", "output_randomwalk"):
            self.output_randomwalk = data_path + \
                conf.get("Data_Out", "output_randomwalk")

        if conf.has_option("Data_Out", "output_embfold"):
            self.output_embfold = data_path + \
                conf.get("Data_Out", "output_embfold")
        else:
            self.output_embfold = False

        if conf.has_option("Data_Out", "output_modelfold"):
            self.output_modelfold = data_path + \
                conf.get("Data_Out", "output_modelfold")
        else:
            self.output_modelfold = False

        if conf.has_option("Data_Out", "output_datafold"):
            self.output_datafold = data_path + \
                conf.get("Data_Out", "output_datafold")
        else:
            self.output_datafold = False

        if self.model == "RHINE":
            self.data_set = conf.get("Model_Setup", "data_set")
            self.mode = conf.get("Model_Setup", "mode")
            self.work_threads = conf.getint("Model_Setup", "work_threads")
            self.epochs = conf.getint("Model_Setup", "epochs")
            self.IRs_nbatches = conf.getint("Model_Setup", "IRs_nbatches")
            self.ARs_nbatches = conf.getint("Model_Setup", "ARs_nbatches")
            self.alpha = conf.getfloat("Model_Setup", "alpha")
            self.margin = conf.getint("Model_Setup", "margin")
            self.dim = conf.getint("Model_Setup", "dim")
            self.ent_neg_rate = conf.getint("Model_Setup", "ent_neg_rate")
            self.rel_neg_rate = conf.getint("Model_Setup", "rel_neg_rate")
            self.evaluation_flag = conf.get("Model_Setup", "evaluation_flag")
            self.train_times = conf.getint("Model_Setup", "train_times")
            self.log_on = conf.getint("Model_Setup", "log_on")
            self.lr_decay = conf.getfloat("Model_Setup", "lr_decay")
            self.exportName = conf.get("Model_Setup", "exportName")
            if self.exportName == 'None':
                self.importName = None
            self.importName = conf.get("Model_Setup", "importName")
            if self.importName == 'None':
                self.importName = None
            self.export_steps = conf.getint("Model_Setup", "export_steps")
            self.opt_method = conf.get("Model_Setup", "opt_method")
            self.optimizer = conf.get("Model_Setup", "optimizer")
            if self.optimizer == 'None':
                self.optimizer = None
            self.weight_decay = conf.get("Model_Setup", "weight_decay")

        elif self.model == "metapath2vec":
            self.epochs = conf.getint("Model_Setup", "epochs")
            self.learning_rate = conf.getfloat("Model_Setup", "learning_rate")
            self.log_dir = data_path + conf.get("Model_Setup", "log_dir")
            self.log_interval = conf.getint("Model_Setup", "log_interval")
            self.max_keep_model = conf.getint("Model_Setup", "max_keep_model")
            self.dim = conf.getint("Model_Setup", "dim")
            self.negative_samples = conf.getint("Model_Setup", "negative_samples")
            self.care_type = conf.getint("Model_Setup", "care_type")
            self.window = conf.getint("Model_Setup", "window")
            self.walk_times = conf.getint("Model_Setup", "walk_times")
            self.walk_length = conf.getint("Model_Setup", "walk_length")
            self.mp_type = conf.get("Model_Setup", "mp_type")
        elif self.model == "metagraph2vec":
            # TODO
            pass
        elif self.model == "PME":
            # TODO
            pass
        else:
            pass
