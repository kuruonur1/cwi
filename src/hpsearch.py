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



def main(opts):
    import cwi, utils
    parser = cwi.get_arg_parser()
    args = vars(parser.parse_args())
    utils.setup_logger(args)
    dset = utils.get_dset()
    objfunc = partial(cwi.xvalidate, dset, 1)

    hpspace = Space(5,opts)
    logging.critical(hpspace)
    logging.critical('')

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
            max_evals=100
            )
    """
    {'opt': 0, 'dr2m3': 0.4161315610584588, 'activation': 2, 'n_batch': 1, 'h2m3': 0, 'dr0m3': 0.5461767196698459, 'dr1m3': 0.6720178139274229, 'h3m3': 0, 'dr3m3': 0.419643470550215, 'h1m3': 1, 'dpart': 2, 'lr': 0.0008630799496044787, 'norm': 8.679706724218939}
    """
    logging.critical(best)

if __name__ == '__main__':
    hyperopt.base.logger.setLevel(logging.DEBUG)
    OPTS = {
            # 'activation' : ['bi-relu','bi-lrelu','bi-elu','bi-lstm'],
            'activation' : ['bi-gru'],
            'emb' : [128],
            'hidden' : [128,256],
            'n_batch' : [5,10,20,40],
            'opt' : ['adam'],
            'drate' : [.2, .8],
            'lr': [.0001,.1],
            'norm': [0.1,10],
            }
    main(OPTS)
