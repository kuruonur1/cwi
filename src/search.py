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
    print base_log_name

    ihandler = logging.FileHandler('logs/{}.crit'.format(base_log_name), mode='w')
    ihandler.setLevel(logging.CRITICAL)
    logger.addHandler(ihandler);

    dhandler = logging.FileHandler('logs/{}.debug'.format(base_log_name), mode='w')
    dhandler.setLevel(logging.DEBUG)
    logger.addHandler(dhandler);

def get_arg_parser():
    parser = argparse.ArgumentParser(prog="search", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--log", default='auto', help="log file name")
    parser.add_argument("--data", default='training', choices=['training','testing_annotated'], help="log file name")
    parser.add_argument("--evals", default=100, type=int, help="num of configurations to try")
    parser.add_argument("--embs", nargs='+', default=['s50','g50'], help='which embs') 
    parser.add_argument("--e_context", default=0, type=int, help="e context")
    parser.add_argument("--feats", default='psn', help="which features")

    parser.add_argument("--kerngamma", nargs='+', default=['loguniform',-15,5], help='gammas')
    parser.add_argument("--C", nargs='+', default=['loguniform',-15,5], help='C distribution')
    parser.add_argument("--percentile", nargs='+', default=['quniform',5,30,1], help='percentile distribution')

    return parser

def main(args):
    logging.critical(tabulate([args], headers='keys', floatfmt='.2f')+'\n')
    random.seed(0)
    defaults = {'embs': args['embs'], 'n_fold' : 5, 'e_context' : args['e_context'],
            'feats' : args['feats'], 'percentile' : 20, 'unkt' : 2,
            'clf' : 'svm', 'kerntype' : 'rbf', 'kerngamma' : 1, 'kerncoef0' : 1, 'kerndegree' : 2, 'cweights' : 1}

    
    assert hasattr(hp,'loguniform')
    space = {
            # 'kerngamma' : hp.loguniform('kerngamma', -20, 8),
            'kerngamma' : getattr(hp, args['kerngamma'][0])('kerngamma', *map(int, args['kerngamma'][1:])),
            'C' : getattr(hp,args['C'][0])('C', *map(int, args['C'][1:])),
            'percentile' : getattr(hp,args['percentile'][0])('percentile', *map(int, args['percentile'][1:])),
            # 'percentile' : hp.quniform('percentile', 5, 30,1),
            # 'percentile' : getattr(hp,args['percentile'][0])('percentile',args['percentile'][1], args['percentile'][2]),
            # 'C' : hp.loguniform('C', -20, 8),
            # 'C' : hp.lognormal('C', -4, 1),
            # 'percentile' : hp.normal('percentile', 20, 5),
            # 'kerngamma' : hp.uniform('kerngamma', .00001, .1),
    }

    from cwi import xvalidate, Emb
    import utils
    dset = utils.get_dset(args['data'])
    emb = Emb(dset)


    def objwrapper(spsample):
        conf = defaults.copy()
        conf.update(spsample)
        logging.critical(tabulate([conf], headers='keys', floatfmt='.2e'))
        f1, f1std = xvalidate(dset, conf, emb)
        loss = 1-f1
        logging.critical('f1:{:.2f} f1std:{:.2f}\n'.format(f1,f1std))

        return {'loss': loss, 'status': STATUS_OK}


    best = fmin(objwrapper,
            space=space,
            algo=tpe.suggest,
            max_evals=args['evals']
            )
    logging.critical(best)
    defaults.update(best)
    f1, f1std = xvalidate(dset, defaults, emb)
    logging.critical('f1:{:.2f} f1std:{:.2f}\n'.format(f1,f1std))

if __name__ == '__main__':
    parser = get_arg_parser()
    args = vars(parser.parse_args())
    setup_logger(args)
    main(args)
