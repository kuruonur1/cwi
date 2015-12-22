from functools import partial
from tabulate import tabulate

import logging
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import utils

class Space(object):

    def __init__(self, max_layerc, opts):
        self.opts = opts
        self.max_layerc = max_layerc
        dpart = [
            dict(
                [('h%dm%d'%(l,maxl), hp.choice('h%dm%d'%(l,maxl), opts['hidden'])) for l in range(1,maxl+1)] +
                [('dr%dm%d'%(l,maxl), hp.uniform('dr%dm%d'%(l,maxl), *opts['drate'])) for l in range(0,maxl+1)]
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
        # self.params = {'rnn':'lazrnn','fepoch':10}
        self.params = args
        self.params.update(spsample)
        self.params['n_hidden'] = map(lambda x:x[1], sorted((k,v) for k, v in spsample['dpart'].iteritems() if k.startswith('h')))
        self.params['drates'] = map(lambda x:x[1], sorted((k,v) for k, v in spsample['dpart'].iteritems() if k.startswith('dr')))

    def __repr__(self):
        keys = ['opt', 'activation', 'drates', 'n_batch', 'lr', 'fepoch', 'n_hidden', 'norm']
        return tabulate([map(self.params.get,keys)], headers=keys)

def setup_logger():
    import socket
    host = socket.gethostname().split('.')[0]
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    shandler = logging.StreamHandler()
    shandler.setLevel(logging.CRITICAL)
    logger.addHandler(shandler);

    """
    if len(args['log']) > 0 and args['log'] != 'nothing':
        ihandler = logging.FileHandler('{}.log'.format(args['log']), mode='w')
        ihandler.setLevel(logging.DEBUG)
        logger.addHandler(ihandler);
    """


def main(opts):
    import cwi
    parser = cwi.get_arg_parser()
    args = vars(parser.parse_args())
    dset = utils.get_dset()
    objfunc = partial(cwi.xvalidate, dset, 4)

    hpspace = Space(3,opts)
    logging.critical(hpspace)

    def objwrapper(spsample):
        conf = Conf(spsample,args)
        logging.critical(conf)
        f1 = objfunc(conf.params)
        loss = 1-f1
        logging.critical('f1:{:.2f}'.format(f1))
        logging.critical('')

        return {'loss': loss, 'status': STATUS_OK}


    best = fmin(objwrapper,
            space=hpspace.space,
            algo=tpe.suggest,
            max_evals=10
            )
    logging.critical(best)

if __name__ == '__main__':
    setup_logger()
    hyperopt.base.logger.setLevel(logging.DEBUG)
    OPTS = {
            'activation' : ['bi-relu','bi-lrelu','bi-elu'],
            'hidden' : [128,256],
            'n_batch' : [32,64,128,256],
            'opt' : ['adam'],
            'drate' : [.2, .8],
            'lr': [.0001,.001],
            'norm': [0.1,100],
            }
    main(OPTS)
