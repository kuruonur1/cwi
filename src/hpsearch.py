from functools import partial
from tabulate import tabulate

import random
import logging
import lasagne
import hyperopt
import argparse
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import utils

class Space(object):

    def __init__(self, max_layerc, opts):
        self.opts = opts
        self.max_layerc = max_layerc
        dpart = [
            dict(
                [('h%dm%d'%(l,maxl), hp.choice('h%dm%d'%(l,maxl), opts['hidden'])) for l in range(1,maxl+1)] +
                [('dr%dm%d'%(l,maxl), hp.choice('dr%dm%d'%(l,maxl), opts['drate'])) for l in range(0,maxl+1)]
            ) for maxl in range(1,self.max_layerc+1)]

        self.space = {
                'activation' : hp.choice('activation', opts['activation']),
                'n_batch':hp.choice('n_batch', opts['n_batch'] ),
                'opt':hp.choice('opt', opts['opt']),
                'lr':hp.uniform('lr', *opts['lr']),
                'norm':hp.uniform('norm', *opts['norm']),
                'dpart' : hp.choice('dpart', dpart),
        }

    def __repr__(self):
        return tabulate([self.opts], headers='keys')

    def sample(self):
        import hyperopt.pyll.stochastic
        return hyperopt.pyll.stochastic.sample(self.space)


class Conf(object):

    def __init__(self, spsample, args):
        self.pnames = ['activation','n_batch','opt','lr','norm']
        self.params={'activation':'bi-relu','n_hidden':[128],'fbmerge':'concat','drates':[0,0],'opt':'adam',\
                'lr':.01,'norm':1,'gclip':0,'truncate':-1,'in2out':0,'emb':0,'fbias':0,'rnn':args['rnn'],'fepoch':args['fepoch']}
        # self.params = {'rnn':'lazrnn','fepoch':10}
        # self.params = args
        self.params.update(spsample)
        self.params['n_hidden'] = map(lambda x:x[1], sorted((k,v) for k, v in spsample['dpart'].iteritems() if k.startswith('h')))
        self.params['drates'] = map(lambda x:x[1], sorted((k,v) for k, v in spsample['dpart'].iteritems() if k.startswith('dr')))

    def __repr__(self):
        keys = ['opt', 'activation', 'drates', 'n_batch', 'lr', 'fepoch', 'n_hidden', 'norm']
        return tabulate([map(self.params.get,keys)], headers=keys)

def setup_logger(args):
    import socket
    host = socket.gethostname().split('.')[0]
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    shandler = logging.StreamHandler()
    shandler.setLevel(logging.CRITICAL)
    logger.addHandler(shandler);

    if len(args['log']) > 0 and args['log'] != 'nothing':
        ihandler = logging.FileHandler('logs/{}.crit'.format(args['log']), mode='w')
        ihandler.setLevel(logging.CRITICAL)
        logger.addHandler(ihandler);

        dhandler = logging.FileHandler('logs/{}.debug'.format(args['log']), mode='w')
        dhandler.setLevel(logging.DEBUG)
        logger.addHandler(dhandler);

def get_arg_parser():
    parser = argparse.ArgumentParser(prog="hpcwi")
    
    parser.add_argument("--rnn", default='lazrnn', choices=['lazrnn','dummy'], help="how to merge forward backward layer outputs")
    parser.add_argument("--activations", default=['bi-gru'], nargs='+', help='activation function for hidden layer: bi-relu bi-lstm bi-tanh')
    parser.add_argument("--opts", default=['adam'], nargs='+', help="opt algos")
    parser.add_argument("--fepoch", default=20, type=int, help="number of epochs")
    parser.add_argument("--evals", default=100, type=int, help="num of configurations to try")
    parser.add_argument("--kfold", default=1, type=int, help="num of folds in xvalidate")
    parser.add_argument("--layers_max", default=3, type=int, help="max num of layers to try")
    parser.add_argument("--log", default='nothing', help="log file name")

    return parser

def main(opts, args):
    random.seed(0)
    rng = np.random.RandomState(1234567)
    lasagne.random.set_rng(rng)
    from cwi import xvalidate
    dset = utils.get_dset()
    objfunc = partial(xvalidate, dset, args['kfold'])

    hpspace = Space(args['layers_max'],opts)
    logging.critical('""" space """')
    logging.critical(hpspace)
    logging.critical('""" space end """\n')

    def objwrapper(spsample):
        conf = Conf(spsample,args)
        logging.critical(conf)
        f1 = objfunc(conf.params)
        loss = 1-f1
        logging.critical('f1:{:.2f}\n'.format(f1))

        return {'loss': loss, 'status': STATUS_OK}


    best = fmin(objwrapper,
            space=hpspace.space,
            algo=tpe.suggest,
            max_evals=args['evals']
            )
    """
    {'opt': 0, 'dr2m3': 0.4161315610584588, 'activation': 2, 'n_batch': 1, 'h2m3': 0, 'dr0m3': 0.5461767196698459, 'dr1m3': 0.6720178139274229, 'h3m3': 0, 'dr3m3': 0.419643470550215, 'h1m3': 1, 'dpart': 2, 'lr': 0.0008630799496044787, 'norm': 8.679706724218939}
    """
    logging.critical(best)

if __name__ == '__main__':
    parser = get_arg_parser()
    args = vars(parser.parse_args())
    setup_logger(args)
    logging.critical('""" args """')
    for k,v in sorted(args.iteritems()):
        logging.critical('{}:{}'.format(k,v))
    logging.critical('""" args end """\n')
    hyperopt.base.logger.setLevel(logging.DEBUG)
    OPTS = {
            # 'activation' : ['bi-relu','bi-lrelu','bi-elu','bi-lstm'],
            'activation' : args['activations'],
            'emb' : [128],
            'hidden' : [128,256],
            'n_batch' : [5,10,20,40],
            'opt' : args['opts'],
            # 'drate' : [.2, .8],
            'drate' : [.2, .3, .4, .5, .6, .7, .8],
            'lr': [.0001,.1],
            'norm': [0.1,10],
            }
    main(OPTS, args)
