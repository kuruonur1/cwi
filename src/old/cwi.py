from __future__ import division

import random
import theano
import logging
import lasagne
import argparse
import numpy as np

from itertools import *
from tabulate import tabulate
from collections import Counter

import utils
import featchar
import evaluate_system
from lazrnn import RDNN, RDNN_Dummy

def get_arg_parser():
    parser = argparse.ArgumentParser(prog="cwi")
    
    parser.add_argument("--rnn", default='lazrnn', choices=['lazrnn','dummy'], help="how to merge forward backward layer outputs")
    parser.add_argument("--activation", default='bi-relu', help="activation function for hidden layer: bi-relu bi-lstm bi-tanh")
    parser.add_argument("--n_hidden", default=[128], nargs='+', type=int, help="number of neurons in each hidden layer")
    parser.add_argument("--drates", default=[0, 0], nargs='+', type=float, help="dropout rates")
    parser.add_argument("--fbmerge", default='concat', choices=['concat','sum'], help="how to merge forward backward layer outputs")
    parser.add_argument("--opt", default="adam", help="optimization method: sgd, rmsprop, adagrad, adam")
    parser.add_argument("--lr", default=.01, type=float, help="learning rate")
    parser.add_argument("--norm", default=2, type=float, help="Threshold for clipping norm of gradient")
    parser.add_argument("--emb", default=0, type=int, help="embedding layer size")
    parser.add_argument("--gclip", default=0, type=float, help="clip gradient messages in recurrent layers if they are above this value")
    parser.add_argument("--truncate", default=-1, type=int, help="backward step size")
    parser.add_argument("--in2out", default=0, type=int, help="connect input & output")
    parser.add_argument("--fbias", default=0., type=float, help="forget gate bias")
    parser.add_argument("--eps", default=1e-8, type=float, help="epsilon for adam")
    parser.add_argument("--gnoise", default=False, action='store_true', help="adding time dependent noise to the gradients")

    parser.add_argument("--fepoch", default=20, type=int, help="number of epochs")
    parser.add_argument("--patience", default=-1, type=int, help="how patient the validator is")

    parser.add_argument("--n_batch", default=5, type=int, help="batch size")
    parser.add_argument("--sample", default=0, type=int, help="num of sents to sample from trn in the order of K")
    parser.add_argument("--shuf", default=1, type=int, help="shuffle the batches.")
    parser.add_argument("--sorted", default=1, type=int, help="sort datasets before training and prediction")

    parser.add_argument("--save", default=False, action='store_true', help="save param values to file")
    parser.add_argument("--log", default='nothing', help="log file name")

    return parser


class Batcher(object):

    def __init__(self, batch_size, feat):
        self.batch_size = batch_size
        self.feat = feat

    def get_batches(self, dset):
        nf = self.feat.NF 
        sent_batches = [dset[i:i+self.batch_size] for i in range(0, len(dset), self.batch_size)]
        X_batches, Xmsk_batches, y_batches, ymsk_batches = [], [], [], []
        batches = []
        for batch in sent_batches:
            mlen = max(len(sent['cseq']) for sent in batch)
            X_batch = np.zeros((len(batch), mlen, nf),dtype=theano.config.floatX)
            Xmsk_batch = np.zeros((len(batch), mlen),dtype=np.bool)
            y_batch = np.zeros((len(batch), mlen, 2),dtype=theano.config.floatX)
            ymsk_batch = np.zeros((len(batch), mlen, 2),dtype=np.bool)
            for si, sent in enumerate(batch):
                Xsent, ysent = self.feat.transform(sent)
                nchar = Xsent.shape[0]
                X_batch[si,:nchar,:] = Xsent
                Xmsk_batch[si,:nchar] = True
                y_batch[si,:nchar,:] = ysent
                ymsk_batch[si,:nchar,:] = True
            batches.append((X_batch, Xmsk_batch, y_batch, ymsk_batch))
        return batches

    def get_batches_sigmoid(self, dset):
        nf = self.feat.NF 
        sent_batches = [dset[i:i+self.batch_size] for i in range(0, len(dset), self.batch_size)]
        X_batches, Xmsk_batches, y_batches, ymsk_batches = [], [], [], []
        batches = []
        for batch in sent_batches:
            mlen = max(len(sent['cseq']) for sent in batch)
            X_batch = np.zeros((len(batch), mlen, nf),dtype=theano.config.floatX)
            Xmsk_batch = np.zeros((len(batch), mlen),dtype=np.bool)
            y_batch = np.zeros((len(batch), mlen),dtype=theano.config.floatX)
            ymsk_batch = np.zeros((len(batch), mlen),dtype=np.bool)
            for si, sent in enumerate(batch):
                Xsent, ysent = self.feat.transform(sent)
                nchar = Xsent.shape[0]
                X_batch[si,:nchar,:] = Xsent
                Xmsk_batch[si,:nchar] = True
                y_batch[si,:nchar] = ysent
                ymsk_batch[si,:nchar] = True
            batches.append((X_batch, Xmsk_batch, y_batch, ymsk_batch))
        return batches

def get_ls(wiseq, tseq):
    tgroup = [[e[0] for e in g] for k, g in groupby(enumerate(wiseq),lambda x: x[1])]
    tseqgrp = [[tseq[ti] for ti in ts] for ts in tgroup]
    return [Counter(tseq).most_common(1)[0][0] for tseq in tseqgrp]


def evalu(rdnn, ddat, dset):
    cost, preds = rdnn.predict(ddat)
    preds = [p.argmax(axis=-1).flatten() for b in preds for p in b]
    correct = sum(np.sum(np.array(sent['lseq']) == pred) for sent, pred in zip(dset,preds))
    total = sum(len(p) for p in preds)
    cerr = 1-(correct/total)

    gold_labels = []
    pred_labels = []
    for sent, pred in zip(dset,preds):
        ls = get_ls(sent['wiseq'],pred)
        gold_labels.extend([sent['ls'][ii] for ii, interested in enumerate(sent['ii']) if interested])
        pred_labels.extend([ls[ii] for ii, interested in enumerate(sent['ii']) if interested])
    p, r, f = evaluate_system.evaluateIdentifier(gold_labels, pred_labels)
    return [cost, cerr, p, r, f]

def xvalidate(dset, k, args):
    feat = featchar.Feat()
    feat.fit(dset)
    batcher = Batcher(args['n_batch'], feat)

    rdnn = RDNN_Dummy(2, feat.NF, args) if args['rnn'] == 'dummy' else RDNN(2, feat.NF, args)
    default_param_values = rdnn.get_param_values() # parameter default values
    fold_indxs = [0,40] if k == 1 else range(0,len(dset),int(np.ceil(len(dset)/k))) + [len(dset)]
    fold_scores = []
    for fstart, fend in zip(fold_indxs, fold_indxs[1:]):
        logging.debug('fstart:{} fend:{}'.format(fstart,fend))
        trn = dset[:fstart] + dset[fend:]
        dev = dset[fstart:fend]

        batcher = Batcher(args['n_batch'], feat)
        rdnn.set_param_values(default_param_values)
        f = validate(rdnn, trn, dev, args['fepoch'], batcher)
        fold_scores.append(f)
    logging.debug('mean:{:.2f} std:{:.2f}'.format(np.mean(fold_scores), np.std(fold_scores)))
    return np.mean(fold_scores)

def validate(rdnn, trn, dev, fepoch, batcher):
    trndat = batcher.get_batches(trn)
    devdat = batcher.get_batches(dev)
    f1s = []
    for e in range(1,fepoch+1):
        cost = rdnn.train(trndat)
        if cost < .01: break

        row1 = evalu(rdnn, trndat, trn) # [cost, cerr, p, r, f]
        row2 = evalu(rdnn, devdat, dev)
        rows = [['trn',e] + row1, ['dev',e] + row2]
        logging.debug(tabulate(rows, floatfmt='.2f', headers=['dset','epoch','cost','cerr','p','r','f']))
        if row2[1] < .3: f1s.append(row2[-1])
    return max(f1s) if len(f1s) else 0.

def setup_logger(args):
    import socket
    host = socket.gethostname().split('.')[0]
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    shandler = logging.StreamHandler()
    shandler.setLevel(logging.DEBUG)
    logger.addHandler(shandler);

    if len(args['log']) > 0 and args['log'] != 'nothing':
        ihandler = logging.FileHandler('logs/{}.log'.format(args['log']), mode='w')
        ihandler.setLevel(logging.DEBUG)
        logger.addHandler(ihandler);

def main():
    random.seed(0)
    rng = np.random.RandomState(1234567)
    lasagne.random.set_rng(rng)
    parser = get_arg_parser()
    args = vars(parser.parse_args())

    setup_logger(args)
    dset = utils.get_dset()
    sent_lens = [len(sent['ws']) for sent in dset]
    logging.debug('# of words per sent, min:{} max:{} mean:{:.2f} std:{:.2f}'.format(min(sent_lens), max(sent_lens), np.mean(sent_lens), np.std(sent_lens)))

    logging.critical(tabulate([args], headers='keys'))

    xvalidate(dset, 1, args)

if __name__ == '__main__':
    main()
