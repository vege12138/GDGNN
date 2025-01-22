import os
import datetime
import argparse
import random
import numpy as np
import torch
import logging
import logging.config


class OptInit():
    def __init__(self):
        parser = argparse.ArgumentParser(description='GDNN')

        datasetLS = ["cora", "citeseer", 'amazon_photo', 'wiki', "amac", "coauther_cs", "pubmed",'coauther_phy']
        # base
        parser.add_argument('--if_early_stop', type=bool, default=True, help='Enable early stopping')
        parser.add_argument('--name', type=str, default=datasetLS[3], help='Dataset name')
        parser.add_argument('--epochs', type=int, default=800, help='Number of training epochs')
        parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=5e-5, help='L2 regularization weight')


        parser.add_argument('--num_convs', type=int, default=6, help='Number of graph_convolution layers')
        parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')

        parser.add_argument('--decay_factor', type=float, default=1.0, help='Decay factor for learning rate')

        parser.add_argument('--oriRes', type=float, default=0.1, help='Initial residual value')

        parser.add_argument('--dropP', type=float, default=0.5, help='drop out')
        parser.add_argument('--num_experts', type=int, default=5, help='num_experts')


        parser.add_argument('--top_k_experts', type=int, default=3, help='num_experts')
        parser.add_argument('--kl', type=float, default=1e-7, help='Initial residual value')


        parser.add_argument('--use_custom_bias', type=bool, default=False, help='Use custom bias initialization')#True:citeseer

        parser.add_argument('--early_stop', type=float, default=100, help='Early stopping patience')

        parser.add_argument('--seed', type=int, default=3407, help='Random seed')

        parser.add_argument('--device', type=int, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='device')
        parser.add_argument('--phase', default='train', type=str, help='train or test(default)')
        parser.add_argument('--use_cpu', action='store_true', help='use cpu?')

        # dataset args
        parser.add_argument('--data_dir', type=str, default='/data/deepgcn/ppi')

        # train args
        parser.add_argument('--iter', default=-1, type=int, help='number of iteration to start')
        parser.add_argument('--postname', type=str, default='', help='postname of saved file')
        parser.add_argument('--multi_gpus', action='store_true', help='use multi-gpus')

        # model args
        parser.add_argument('--pretrained_model', type=str, help='path to pretrained model(default: none)', default='')
        parser.add_argument('--model_name', type=str, default='')
        # convolution
        # dilated knn

        # saving
        parser.add_argument('--ckpt_path', type=str, default='')
        parser.add_argument('--save_best_only', default=True, type=bool, help='only save best model')

        args = parser.parse_args()
        args.time = datetime.datetime.now().strftime("%y%m%d")

        dir_path = os.path.dirname(os.path.abspath(__file__))
        args.task = os.path.basename(dir_path)
        args.post = '-'.join([args.task])
        if args.postname:
            args.post += '-' + args.postname


        if args.pretrained_model:
            if args.pretrained_model[0] != '/':
                if args.pretrained_model[0:2] == 'ex':
                    args.pretrained_model = os.path.join(os.path.dirname(os.path.dirname(dir_path)),
                                                         args.pretrained_model)
                else:
                    args.pretrained_model = os.path.join(dir_path, args.pretrained_model)
                args.pretrained_model = os.path.join(dir_path, args.pretrained_model)

        if not args.ckpt_path:
            args.save_path = os.path.join(dir_path, 'checkpoints/ckpts'+'-'+args.post + '-' + args.time)
        else:
            args.save_path = os.path.join(args.ckpt_path, 'checkpoints/ckpts' + '-' + args.post + '-' + args.time)

        args.logdir = os.path.join(dir_path, 'logs/'+args.post + '-' + args.time)

        if args.use_cpu:
            args.device = torch.device('cpu')
        else:
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args = args

    def initialize(self):
        if self.args.phase=='train':
            # logger
            # self.args.logger = TfLogger(self.args.logdir)
            # loss
            self.args.epoch = -1
            self.make_dir()

        self.set_seed(3407)
        self.logging_init()
        self.print_args()
        return self.args

    def print_args(self):
        # self.args.printer args
        self.args.printer.info("==========       CONFIG      =============")
        cnt = 0
        print()
        self.args.printer.info("run_time:{}".format(self.args.time))
        for arg, content in self.args.__dict__.items():
            if cnt > 14:
                break
            self.args.printer.info("{}:{}".format(arg, content))
            cnt+=1
        self.args.printer.info("==========     CONFIG END    =============")
        self.args.printer.info("\n")
        self.args.printer.info('===> Phase is {}.'.format(self.args.phase))

    def logging_init(self):
        if not os.path.exists(self.args.logdir):
            os.makedirs(self.args.logdir)
        ERROR_FORMAT = "%(message)s"
        DEBUG_FORMAT = "%(message)s"
        LOG_CONFIG = {'version': 1,
                      'formatters': {'error': {'format': ERROR_FORMAT},
                                     'debug': {'format': DEBUG_FORMAT}},
                      'handlers': {'console': {'class': 'logging.StreamHandler',
                                               'formatter': 'debug',
                                               'level': logging.DEBUG},
                                   'file': {'class': 'logging.FileHandler',
                                            'filename': os.path.join(self.args.logdir, self.args.post+'.log'),
                                            'formatter': 'debug',
                                            'level': logging.DEBUG}},
                      'root': {'handlers': ('console', 'file'), 'level': 'DEBUG'}
                      }
        logging.config.dictConfig(LOG_CONFIG)
        self.args.printer = logging.getLogger(__name__)

    def make_dir(self):
        # check for folders existence
        if not os.path.exists(self.args.logdir):
            os.makedirs(self.args.logdir)
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        if not os.path.exists(self.args.data_dir):
            os.makedirs(self.args.data_dir)

    def set_seed(self, seed=3407):
# 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


