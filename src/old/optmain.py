import lasagne, theano, numpy as np, logging
from theano import tensor as T
import argparse, os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt.pyll.stochastic
from tabulate import tabulate
from collections import defaultdict as dd

import model
import prep

# random.seed(0)
rng = np.random.RandomState(1234567)
lasagne.random.set_rng(rng)

# param_names = ['activation','n_hidden','drates','opt','lr','norm']
def get_arg_parser():
    parser = argparse.ArgumentParser(prog="main")
    
    parser.add_argument("--toy", default=1, type=int, help="toy or real")
    parser.add_argument("--hepoch", default=10, type=int, help="num of epochs")
    parser.add_argument("--fepoch", default=100, type=int, help="num of epochs")
    parser.add_argument("--max_evals", default=100, type=int, help="num of configs to try")
    parser.add_argument("--max_layers", default=3, type=int, help="num of configs to try")
    parser.add_argument("--log", default='nothing', help="log file name")

    return parser

def setup_logger(args):
    import socket
    host = socket.gethostname().split('.')[0]
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    shandler = logging.StreamHandler()
    shandler.setLevel(logging.CRITICAL)
    logger.addHandler(shandler);

    if len(args['log']) > 0 and args['log'] != 'nothing':
        ihandler = logging.FileHandler('{}.log'.format(args['log']), mode='w')
        ihandler.setLevel(logging.INFO)
        logger.addHandler(ihandler);


def create_space(max_layer_count, opts):
    MAXL=max_layer_count
    dpart = [
        dict(
            [('h%dm%d'%(l,maxl), hp.choice('h%dm%d'%(l,maxl), opts['hidden'])) for l in range(1,maxl+1)] +
            [('d%dm%d'%(l,maxl), hp.uniform('d%dm%d'%(l,maxl), .1,.9)) for l in range(0,maxl+1)]
        ) for maxl in range(1,MAXL+1)]

    space = {
            'activation' : hp.choice('activation', opts['activation']),
            'lr':hp.uniform('lr',.0001,.001),
            'norm':hp.uniform('norm',0.1,100),
            'bnorm':hp.choice('bnorm', opts['bnorm']),
            'n_batch':hp.choice('n_batch', opts['n_batch']),
            'opt':hp.choice('opt', opts['opt']),
            'dpart' : hp.choice('dpart', dpart),
    }
    return space


def main():
    parser = get_arg_parser()
    args = vars(parser.parse_args())
    setup_logger(args)

    logging.critical(tabulate([args],headers='keys',tablefmt='plain'))

    OPTS = {
            'activation' : ['sigmoid','tanh','relu', 'elu'],
            'opt' : ['adam'],
            'n_batch' : [32,64,128,256],
            'hidden' : [128,256],
            'bnorm' : [0,1],
            # 'n_batch' : [128,256,512],
            # 'hidden' : [512,1024],
            }

    NF, NOUT = 400, 200
    logging.critical('loading data...')
    if args['toy']:
        dat = np.load('toy.npz')
        trn, dev, tst = dat['trn'], dat['dev'], dat['tst']
    else:
        trn, dev, tst = map(prep.get_dset, ('trn','dev','tst'))
        OPTS['n_batch'] = [128,256,512]
        OPTS['hidden'] = [512,1024,2048]

    logging.critical('loading data done.')
    logging.critical(tabulate([OPTS],headers='keys'))
    logging.critical('')

    trnX, trnY = trn[:,NOUT:], trn[:,:NOUT]
    devX, devY = dev[:,NOUT:], dev[:,:NOUT]

    def objective(conf):
        conf['n_hidden'] = map(lambda x:x[1], sorted((k,v) for k, v in conf['dpart'].iteritems() if k.startswith('h')))
        conf['drates'] = map(lambda x:x[1], sorted((k,v) for k, v in conf['dpart'].iteritems() if k.startswith('d')))

        dnn = model.DNN(NF,NOUT,conf)

        dcosts = []
        for e in range(args['hepoch']):
            tcost = dnn.train(trnX, trnY)
            dcost, pred = dnn.predict(devX, devY)
            dcosts.append(dcost)

        dcost = min(dcosts)
        dcost = np.iinfo(np.int32).max if np.isnan(dcost) else dcost

        info = dd(lambda : None)
        info.update(conf)
        # info = conf.copy()
        info['loss'] = dcost
        info.update(('h%d'%i,nh) for i, nh in enumerate(info['n_hidden'],1))
        info.update(('dr%d'%i,dr) for i, dr in enumerate(info['drates']))
        # map(info.pop, ('dpart','n_hidden','drates'))
        headers=['loss','n_batch','opt','activation','lr','norm', 'bnorm']+ ['h%d'%i for i in range(1,args['max_layers']+1)]+ ['dr%d'%i for i in range(args['max_layers']+1)]
        logging.critical(tabulate([map(lambda x:info[x], headers)],headers=headers, floatfmt='.4f'))

        return {
                'loss': dcost,
                'status': STATUS_OK,}

    space = create_space(args['max_layers'], OPTS)

    best = fmin(objective,
            space=space,
            algo=tpe.suggest,
            max_evals=args['max_evals']
            )
    logging.critical(best)
    logging.critical('')
    best_params = best2mparams(best, OPTS)
    logging.critical(tabulate([best_params], headers='keys'))

    dnn = model.DNN(NF,NOUT,best_params)
    for e in range(args['fepoch']):
        tcost = dnn.train(trnX, trnY)
        dcost = dnn.predict(devX, devY)

    logging.critical('dcost with best model: {}'.format(dcost))


def best2mparams(best, opts):
    mparams = {}
    mparams.update((e, opts[e][best[e]]) for e in ['activation', 'n_batch', 'opt'])

    mparams.update((e, best[e]) for e in ['norm','lr'])

    n_hidden_ind = map(lambda x:x[1], sorted((k,v) for k, v in best.iteritems() if k.startswith('h')))
    mparams['n_hidden'] = map(lambda i: opts['hidden'][i], n_hidden_ind)
    mparams['drates'] = map(lambda x:x[1], sorted((k,v) for k, v in best.iteritems() if k.startswith('d')))
    return mparams

def hp_test():
    import hyperopt.pyll.stochastic
    space = create_space(3, OPTS)
    print space['activation'][2]
    for e in range(5):
        space_sample = hyperopt.pyll.stochastic.sample(space)
        print space_sample 

if __name__ == '__main__':
    main()

