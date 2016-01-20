from __future__ import division

import random
import logging
import argparse
import numpy as np

from itertools import *
from tabulate import tabulate
from collections import Counter

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier, Perceptron, LogisticRegression
from sklearn.feature_extraction import DictVectorizer

import utils
import evaluate_system

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


def get_arg_parser():
    parser = argparse.ArgumentParser(prog="cwi")
    
    parser.add_argument("--emb", default='', help="how to merge forward backward layer outputs")
    parser.add_argument("--feats", default='wsW', help="which feats to use")
    parser.add_argument("--n_fold", default=5, type=int, help="batch size")
    parser.add_argument("--n_context", default=2, type=int, help="num of words around token")
    parser.add_argument("--fepoch", default=100, type=int, help="number of epochs")
    parser.add_argument("--patience", default=10, type=int, help="angerrr")
    parser.add_argument("--log", default='nothing', help="log file name")
    parser.add_argument("--percentile", default=20, type=int, help="percentile for feature selection")
    parser.add_argument("--cweights", default=1, type=int, help="use cweights")
    parser.add_argument("--unkt", default=2, type=int, help="unk threshold")

    parser.add_argument("--clf", default='svm', choices=['svm','lo'], help="clf type")
    parser.add_argument("--C", default=1, type=float, help="C for SVC")
    parser.add_argument("--kerntype", default='lin', choices=['lin','poly','rbf'],  help="kernel type")
    parser.add_argument("--kerngamma", default=1., type=float, help="poly or rbf gamma")
    parser.add_argument("--kerncoef0", default=1., type=float, help="poly kernel coef0")
    parser.add_argument("--kerndegree", default=2, type=int, help="poly kernel degree")

    parser.add_argument("--testf", default=False, help="print test data")

    return parser

def xvalidate(dset, args):
    fold_indxs = [0,40] if args['n_fold'] == 1 else range(0,len(dset),int(np.ceil(len(dset)/args['n_fold']))) + [len(dset)]
    fold_scores = []
    for fstart, fend in zip(fold_indxs, fold_indxs[1:]):
        logging.debug('fstart:{} fend:{}'.format(fstart,fend))
        trn = dset[:fstart] + dset[fend:]
        dev = dset[fstart:fend]

        f = validate(trn, dev, args)
        fold_scores.append(f)
    f1mean, f1std = np.mean(fold_scores), np.std(fold_scores)
    logging.debug('mean:{:.2f} std:{:.2f}'.format(f1mean, f1std))
    return f1mean, f1std

def validate(trn, dev, args):
    yhat_trn, yhat_dev = predict(trn, dev, args)
    row1 = list(evalu(trn, yhat_trn)) # p, r, f

    row2 = list(evalu(dev, yhat_dev)) # p, r, f
    rows = [['trn'] + row1, ['dev'] + row2]
    logging.debug(tabulate(rows, floatfmt='.2f', headers=['dset','wacc','p','r','f']))

    f1=row2[-1]
    return f1

def predict(trn, dev, args):
    feat = Feat(trn, unkt=args['unkt'], feats=args['feats'])
    dvec = DictVectorizer(sparse=False)
    # dvec.fit(feat.get_features(i, sent) for sent in trn for i,w in enumerate(sent['ws']))

    Xtrn = dvec.fit_transform(feat.get_features(i, sent) for sent in trn for m,(i,w) in zip(sent['ii'],enumerate(sent['ws'])) if m)
    Xdev = dvec.transform(feat.get_features(i, sent) for sent in dev for m,(i,w) in zip(sent['ii'],enumerate(sent['ws'])) if m)
    ytrn = np.array([lbl for sent in trn for m,lbl in zip(sent['ii'],sent['ls']) if m])
    ydev = np.array([lbl for sent in dev for m,lbl in zip(sent['ii'],sent['ls']) if m])
    # cweights = {0:(ytrn==1).sum()/ytrn.size, 1:(ytrn==0).sum()/ytrn.size} if args['cweights'] else {0:1,1:1}
    cweights = 'balanced' if args['cweights'] else {0:1,1:1}
    logging.debug('Xtrn shape:{}'.format(Xtrn.shape))
    logging.debug('Xdev shape:{}'.format(Xdev.shape))


    assert (Xtrn.std(axis=0)==0).sum() == 0

    from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, SelectPercentile

    sel = SelectPercentile(chi2, percentile=args['percentile'])
    Xtrn, Xdev = sel.fit_transform(Xtrn, ytrn), sel.transform(Xdev)

    logging.debug('after sel: Xtrn:{} Xdev: {}'.format(Xtrn.shape, Xdev.shape))
    logging.debug([(fea, score) for score, fea in islice(reversed(sorted(zip(sel.scores_, dvec.feature_names_))), 100)])

    if args['emb']:
        Xtrn_emb = np.array([emb.get_context(i,sent) for sent in trn for m,(i,w) in zip(sent['ii'],enumerate(sent['ws'])) if m])
        Xdev_emb = np.array([emb.get_context(i,sent) for sent in dev for m,(i,w) in zip(sent['ii'],enumerate(sent['ws'])) if m])
        logging.debug('embs: Xtrn_emb:{} Xdev_emb: {}'.format(Xtrn_emb.shape, Xdev_emb.shape))
        Xtrn = np.hstack((Xtrn_emb,Xtrn))
        Xdev = np.hstack((Xdev_emb,Xdev))
        logging.debug('after emb: Xtrn:{} Xdev: {}'.format(Xtrn.shape, Xdev.shape))

    if args['clf'] == 'svm':
        if args['kerntype'] == 'lin':
            clf = LinearSVC(C=args['C'], class_weight=cweights)
        else:
            clf = SVC(class_weight=cweights, C=args['C'], kernel=args['kerntype'], gamma=args['kerngamma'], degree=args['kerndegree'])
    elif args['clf'] == 'lo':
        clf = LogisticRegression(C=args['C'], class_weight=cweights)

    clf.fit(Xtrn, ytrn)

    return clf.predict(Xtrn), clf.predict(Xdev)

class Emb(object):
    
    def __init__(self, fname, dset):
        self.vocab = Counter(w for sent in dset for w in sent['ws'])
        # with open('/ai/home/vcirik/embeddings/wikipedia2MUNK-25.embeddings') as src:
        fname = '/ai/home/vcirik/embeddings/{}'.format(fname)
        logging.debug(fname)
        with open(fname) as src:
            srcgen = (l.strip().split(' ') for l in src)
            self.d = dict((l[0], map(float,l[1:])) for l in srcgen if l[0] in self.vocab or l[0] == '*UNKNOWN*')

        """
        with open('/ai/home/vcirik/embeddings/wikipedia2MUNK-100.embeddings') as src:
            srcgen = (l.strip().split(' ') for l in src)
            self.d2 = dict((l[0], map(float,l[1:])) for l in srcgen if l[0] in vocab or l[0] == '*UNKNOWN*')
        assert '*UNKNOWN*' in self.d
        """
        self.embdim = len(self.d['*UNKNOWN*'])

    def get_w(self, w):
        if w == '<s>' or w == '</s>':
            return [0]*self.embdim
        else:
            emb1 = self.d[w] if w in self.d else self.d['*UNKNOWN*']
            return emb1
            # emb2 = self.d[w] if w in self.d else self.d['*UNKNOWN*']
            # return emb1+emb2

    def get_token(self, i, sent):
        return self.get_w(sent['ws'][i])

    def get_context(self, i, sent, c=2):
        ws_padded = (['<s>']*c) + sent['ws'] + (['</s>']*c)
        # logging.debug(ws_padded)
        l = [fi for w in ws_padded[(i):(i+c+c+1)] for fi in self.get_w(w)]
        # featd.update(('w%d'%j, self.get_w(ws_padded[i+knn+j])) for j in range(-knn,0) + range(1,knn+1))
        if len(l) == 0:
            print 'debug info'
            print l
            print i
            print ws_padded
            assert False
        return l

class Feat(object):

    def __init__(self, dset, unkt=2, feats='w'):
        self.wcounts = Counter(w for sent in dset for w in sent['ws'])
        self.subcounts = Counter(w[:j] for sent in dset for w in sent['ws'] for j in range(1,len(w)-1))
        self.unkt = unkt
        self.feats = feats
        # self.emb = emb

    def get_w(self, w):
        return w if self.wcounts[w] > self.unkt else '<unk>'

    def get_sub(self, sub):
        return sub if self.subcounts[sub] > self.unkt else '<unksub>'

    def get_features(self, i, sent, knn=2):
        featd = {}
        w = sent['ws'][i]

        ws_padded = (['<s>']*knn) + sent['ws'] + (['</s>']*knn)
        # featd = dict(('e%d'%fi, fv) for fi, fv in enumerate(fv for w in ws_padded[(i-c):(i+c+1)] for fv in self.emb.get_w(w)))

        if 'w' in self.feats:
            featd['w'] = self.get_w(w)
        # featd['containsHyphen'] = '-' in w
        if 's' in self.feats:
            featd.update(('substr%d'%j, self.get_sub(w[:j])) for j in range(1,len(w)-1))
        if 'W' in self.feats:
            featd.update(('w%d'%j, self.get_w(ws_padded[i+knn+j])) for j in range(-knn,0) + range(1,knn+1))
        if 'c' in self.feats:
            featd['cntxt'] = ' '.join([ws_padded[i+knn+j] for j in range(-knn,0) + range(1,knn+1)])
        return featd

    
def evalu(dset, yhat):
    # slens = [len(sent['ls']) for sent in dset]
    slens = [sum(sent['ii']) for sent in dset]
    offset = 0
    preds = []
    for slen in slens:
        preds.append(yhat[offset:offset+slen])
        offset+=slen
    assert len(preds) == len(dset)
    gold_labels = []
    pred_labels = []
    for sent, pred in zip(dset,preds):
        gold_labels.extend([sent['ls'][ii] for ii, interested in enumerate(sent['ii']) if interested])
        # pred_labels.extend([pred[ii] for ii, interested in enumerate(sent['ii']) if interested])
        pred_labels.extend(pred)
    p, r, f = evaluate_system.evaluateIdentifier(gold_labels, pred_labels)
    return p,r,f

# def main():
if __name__ == '__main__':
    random.seed(0)
    np.random.seed(1234567)

    parser = get_arg_parser()
    args = vars(parser.parse_args())

    setup_logger(args)

    for k,v in sorted(args.iteritems()):
        logging.critical('{} {}'.format(k,v))

    dset = utils.get_dset()
    if args['emb']:
        emb = Emb(args['emb'], dset)
    sent_lens = [len(sent['ws']) for sent in dset]
    logging.debug('# of words per sent, min:{} max:{} mean:{:.2f} std:{:.2f}'.format(min(sent_lens), max(sent_lens), np.mean(sent_lens), np.std(sent_lens)))
    sent = dset[0]
    utils.pprint_word(sent)

    if args['testf']:
        trn = utils.get_dset()
        tst = utils.get_test()
        ytrn, ytst = predict(trn, tst, args)
        with open(args['testf'], 'w') as out:
            for y in ytst: out.write('%d\n'%y)
    else:
        xvalidate(dset, args)

