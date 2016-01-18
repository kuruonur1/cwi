from functools import partial
from tabulate import tabulate

import random
import logging
import hyperopt
import argparse
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import utils

def setup_logger(args):
    import socket
    host = socket.gethostname().split('.')[0]
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    shandler = logging.StreamHandler()
    shandler.setLevel(logging.CRITICAL)
    logger.addHandler(shandler);

    param_log_name = ','.join(['{}:{}'.format(p,v) for p, v in sorted(args.iteritems())])
    param_log_name = "".join(i for i in param_log_name if i not in "\"\/ &*?<>|[]()'")
    base_log_name = '{}:{},{}'.format(host, 'cpu', param_log_name if args['log'] == 'auto' else args['log'])

    ihandler = logging.FileHandler('logs/{}.crit'.format(base_log_name), mode='w')
    ihandler.setLevel(logging.CRITICAL)
    logger.addHandler(ihandler);

    dhandler = logging.FileHandler('logs/{}.debug'.format(base_log_name), mode='w')
    dhandler.setLevel(logging.DEBUG)
    logger.addHandler(dhandler);

def get_arg_parser():
    parser = argparse.ArgumentParser(prog="kpsearch")

    parser.add_argument("--log", default='auto', help="log file name")
    parser.add_argument("--evals", default=100, type=int, help="log file name")

    return parser

def main(args):
    logging.critical(tabulate([args], headers='keys', floatfmt='.2f')+'\n')
    random.seed(0)
    defaults = {'emb': 'mikolovWikipedia100.embeddings',
            'n_batch' : 100, 'n_fold' : 1, 'fepoch' : 100, 'anger' : 20, 'n_context' : 2,
            'kerntype' : 'rbf', 'kerngamma' : 1, 'kerncoef0' : 1, 'kerndegree' : 2}

    space = {
            # 'kerngamma' : hp.normal('kerngamma', 5, 2),
            'svmC' : hp.uniform('svmC', 1, 10),
            'kerngamma' : hp.uniform('kerngamma', .00001, .1),
    }

    from kpcwi import xvalidate
    import utils
    dset = utils.get_dset()
    # xvalidate,dset, k, args


    def objwrapper(spsample):
        conf = defaults.copy()
        conf.update(spsample)
        logging.critical(tabulate([conf], headers='keys', floatfmt='.2f'))
        f1, fstd = xvalidate(dset, conf)
        loss = 1-f1
        logging.critical('f1:{:.2f}\n'.format(f1))

        return {'loss': loss, 'status': STATUS_OK}


    best = fmin(objwrapper,
            space=space,
            algo=tpe.suggest,
            max_evals=args['evals']
            )
    logging.critical(best)

if __name__ == '__main__':
    parser = get_arg_parser()
    args = vars(parser.parse_args())
    setup_logger(args)
    main(args)
