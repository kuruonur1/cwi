import logging
import numpy as np
from itertools import *
from tabulate import tabulate

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

def opt(dset, emb, embs, e_context):
    dargs = {'e_context' : 0, 'feats' : '', 'percentile' : 0, 'n_fold' : 5, 
            'cweights' : 1, 'clf' : 'svm', 'kerntype' : 'lin', 'C' : 1, 'kerngamma' : 1, 'kerncoef0' : 1, 'kerndegree' : 1}
    dargs['e_context'] = e_context 
    dargs['embs'] = embs 
    f1,f1std,C = c_opt(dset, emb, dargs)
    return {'e_context' :e_context, 'embs':' '.join(embs), 'f1':f1,'f1std':f1std, 'C':C}

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
    setup_logger({'log':'road'})

    dset = utils.get_dset()

    emb = cwi.Emb(dset)

    infolist = []
    for ec in range(4):
        infolist.extend(opt(dset,emb,['{}{}'.format(e,dim)],ec) for e,dim in product(['s','g'],[50,100,200]))
        # infolist.extend(opt(dset,emb,ename,ec) for ename in [['s50']])

    for ec in range(4):
        infolist.extend(opt(dset,emb,[e1,e2],ec) for e1,e2 in product(['s50','s100','s200'],['g50','g100','g200']))

    logging.critical(tabulate(infolist,headers='keys'))




