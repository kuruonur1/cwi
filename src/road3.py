import copy
import random
import logging
import argparse
import numpy as np
from itertools import *
from tabulate import tabulate
from collections import OrderedDict as od

import cwi
import utils


def setup_logger(args):
    import socket
    host = socket.gethostname().split('.')[0]
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)
    shandler = logging.StreamHandler()
    shandler.setLevel(logging.CRITICAL)
    logger.addHandler(shandler);

    if len(args['log']) > 0 and args['log'] != 'dont':
        ihandler = logging.FileHandler('logs/{}.log'.format(args['log']), mode='w')
        ihandler.setLevel(logging.CRITICAL)
        logger.addHandler(ihandler);

C_powers = [-20,+8]
C_values = np.power(np.zeros(sum(map(abs,C_powers)))+2, np.arange(*C_powers))


def c_opt(dset, emb, dargs):
    logging.critical(tabulate([dargs],headers='keys'))
    f1s = []
    for C in C_values:
        targs = dargs.copy()
        targs['C'] = C
        f1, f1std = cwi.xvalidate(dset, targs, emb)
        f1s.append((f1,f1std,C))
        logging.critical('{}\t{}\t{}'.format(C, f1,f1std))
    logging.critical('\n')
    return max(f1s)

if __name__ == '__main__':
    parser = cwi.get_arg_parser()
    dargs = vars(parser.parse_args())

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='dont')
    parser.add_argument('--feats', default='')
    parser.add_argument('--percentile', default=20, type=int)
    parser.add_argument('--e_context', default=0, type=int)
    args = parser.parse_args()
    """
    print tabulate([dargs],headers='keys')

    setup_logger({'log':dargs['log']})

    logging.info(tabulate([dargs],headers='keys'))

    dset0 = utils.get_dset('training')
    dset = utils.get_dset('testing_annotated')
    emb = cwi.Emb(dset0+dset)

    trn, tst = dset0, dset

    # dargs = {'embs':['s50','g50'], 'e_context' : args.e_context, 'feats' : args.feats, 'percentile' : args.percentile, 'n_fold' : 5, 
            # 'cweights' : 1, 'clf' : 'svm', 'kerntype' : 'lin', 'C' : 1, 'kerngamma' : 1, 'kerncoef0' : 1, 'kerndegree' : 1}
    f1,f1std,C = c_opt(trn, emb, dargs)
    dargs['C'] = C
    p, r, f = cwi.validate(trn, tst, dargs, emb)
    print p,r,f


