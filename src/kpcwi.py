from __future__ import division

import random
import logging
import argparse
import numpy as np

from itertools import *
from tabulate import tabulate
from collections import Counter, OrderedDict

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier, Perceptron, LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix

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

    if len(args['log']) > 0 and args['log'] != 'dont':
        ihandler = logging.FileHandler('logs/{}.log'.format(args['log']), mode='w')
        ihandler.setLevel(logging.DEBUG)
        logger.addHandler(ihandler);


def get_arg_parser():
    parser = argparse.ArgumentParser(prog="cwi")
    
    parser.add_argument("--n_fold", default=5, type=int, help="batch size")

    parser.add_argument("--embs", nargs='+', default=['mikolovWikipedia100.embeddings'], help="embeddings to use")
    parser.add_argument("--e_context", default=1, type=int, help="num of words around token")
    parser.add_argument("--feats", default='', help="which feats to use")
    parser.add_argument("--percentile", default=20, type=int, help="percentile for feature selection")
    parser.add_argument("--w_context", default=2, type=int, help="num of words around token")

    parser.add_argument("--cweights", default=1, type=int, help="use cweights")
    parser.add_argument("--unkt", default=2, type=int, help="unk threshold")

    parser.add_argument("--clf", default='lo', choices=['svm','lo'], help="clf type")
    parser.add_argument("--C", default=1, type=float, help="the C")
    parser.add_argument("--kerntype", default='rbf', choices=['lin','poly','rbf'],  help="kernel type")
    parser.add_argument("--kerngamma", default=1., type=float, help="poly or rbf gamma")
    parser.add_argument("--kerncoef0", default=1., type=float, help="poly kernel coef0")
    parser.add_argument("--kerndegree", default=2, type=int, help="poly kernel degree")

    parser.add_argument("--log", default='dont', help="log file name")
    parser.add_argument("--testf", default=False, help="print test data")

    return parser

def xvalidate(dset, args):
    fold_indxs = [0,40] if args['n_fold'] == 1 else range(0,len(dset),int(np.ceil(len(dset)/args['n_fold']))) + [len(dset)]
    fold_scores = []
    for fstart, fend in zip(fold_indxs, fold_indxs[1:]):
        logging.debug('fstart:{} fend:{}'.format(fstart,fend))
        trn = dset[:fstart] + dset[fend:]
        dev = dset[fstart:fend]

        row = validate(trn, dev, args) # [p, r, f]
        fold_scores.append(row)
    meansarr, stdsarr = np.array(fold_scores).mean(axis=0), np.array(fold_scores).std(axis=0)
    logging.debug(tabulate([meansarr, stdsarr], headers=['p','r','f'],floatfmt='.2f'))

def validate(trn, dev, args):
    yhat_trn, yhat_dev = fit_predict(trn, dev, args)
    row1 = list(evalu(trn, yhat_trn)) # p, r, f

    row2 = list(evalu(dev, yhat_dev)) # p, r, f
    rows = [['trn'] + row1, ['dev'] + row2]
    logging.debug(tabulate(rows, floatfmt='.2f', headers=['dset','p','r','f']))

    return row2

def fit_predict(trn, dev, args):
    Xtrn = np.array([emb.get_context(i,sent) for sent in trn for m,(i,w) in zip(sent['ii'],enumerate(sent['ws'])) if m])
    Xdev = np.array([emb.get_context(i,sent) for sent in dev for m,(i,w) in zip(sent['ii'],enumerate(sent['ws'])) if m])
    ytrn = np.array([lbl for sent in trn for m,lbl in zip(sent['ii'],sent['ls']) if m])
    ydev = np.array([lbl for sent in dev for m,lbl in zip(sent['ii'],sent['ls']) if m])
    logging.debug('embs: Xtrn_emb:{} Xdev_emb: {}'.format(Xtrn.shape, Xdev.shape))

    if len(args['feats']):
        feat = Feat(trn, args)
        dvec = DictVectorizer(sparse=False)
        # dvec.fit(feat.get_features(i, sent) for sent in trn for i,w in enumerate(sent['ws']))

        Xtrn_feat = dvec.fit_transform(feat.get_features(i, sent) for sent in trn for m,(i,w) in zip(sent['ii'],enumerate(sent['ws'])) if m)
        Xdev_feat = dvec.transform(feat.get_features(i, sent) for sent in dev for m,(i,w) in zip(sent['ii'],enumerate(sent['ws'])) if m)
        logging.debug('Xtrn_feat shape:{}'.format(Xtrn_feat.shape))
        logging.debug('Xdev_feat shape:{}'.format(Xdev_feat.shape))


        assert (Xtrn.std(axis=0)==0).sum() == 0

        if args['percentile'] < 100:
            from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, SelectPercentile
            sel = SelectPercentile(chi2, percentile=args['percentile'])
            Xtrn_feat, Xdev_feat = sel.fit_transform(Xtrn_feat, ytrn), sel.transform(Xdev_feat)
            logging.debug('after sel: Xtrnf:{} Xdevf: {}'.format(Xtrn_feat.shape, Xdev_feat.shape))
            logging.debug([(fea, score) for score, fea in islice(reversed(sorted(zip(sel.scores_, dvec.feature_names_))), 100)])

        Xtrn = np.hstack((Xtrn,Xtrn_feat))
        Xdev = np.hstack((Xdev,Xdev_feat))
        logging.debug('after feats: Xtrn:{} Xdev: {}'.format(Xtrn.shape, Xdev.shape))


    cweights = 'balanced' if args['cweights'] else {0:1,1:1}
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
    
    def __init__(self, dset, args):
        self.vocab = Counter(w for sent in dset for w in sent['ws'])
        self.n_context = args['e_context']
        self.embsdict = dict((fn,self.read_emb_file(fn)) for fn in args['embs'])
        self.embdim = sum(len(embd['*UNKNOWN*']) for embd in self.embsdict.itervalues())

    def read_emb_file(self, fname):
        fname = '{}/{}'.format(utils.EMB_DIR,fname)
        logging.debug(fname)
        with open(fname) as src:
            srcgen = (l.strip().split(' ') for l in src)
            d = dict((l[0], map(float,l[1:])) for l in srcgen if l[0] in self.vocab or l[0] == '*UNKNOWN*')
        return d

    def get_w(self, w):
        if w == '<s>' or w == '</s>':
            return [0] * self.embdim
        else:
            emb = []
            for fn, embd in self.embsdict.iteritems():
                emb.extend(embd[w] if w in embd else embd['*UNKNOWN*'])
            return emb

    def get_token(self, i, sent):
        return self.get_w(sent['ws'][i])

    def get_context(self, i, sent):
        c = self.n_context
        ws_padded = (['<s>']*c) + sent['ws'] + (['</s>']*c)
        # logging.debug(ws_padded)
        return [fi for w in ws_padded[i:i+c] + [ws_padded[i+c]] + ws_padded[i+c+1:i+c+c+1] for fi in self.get_w(w)]
        """
        l = [fi for w in ws_padded[(i):(i+c+c+1)] for fi in self.get_w(w)]
        # featd.update(('w%d'%j, self.get_w(ws_padded[i+knn+j])) for j in range(-knn,0) + range(1,knn+1))
        if len(l) == 0:
            print 'debug info'
            print l
            print i
            print ws_padded
            assert False
        return l
        """

class Feat(object):

    def __init__(self, dset, args):
        self.wcounts = Counter(w for sent in dset for w in sent['ws'])
        self.precounts = Counter(w[:j] for sent in dset for w in sent['ws'] for j in range(1,len(w)-1))
        self.sufcounts = Counter(w[-j:] for sent in dset for w in sent['ws'] for j in range(1,len(w)-1))
        self.unkt = args['unkt']
        self.feats = args['feats']
        self.n_context = args['w_context']
        # self_ngrams = lambda w,n : 
        # self.emb = emb

    def f_ngrams(self, w, n):
        # return [w[i:i+n] for i in xrange(len(w)-n+1)]
        return [w[i:i+n] for i in xrange(1,len(w)-n)]

    def get_w(self, w):
        return w if self.wcounts[w] > self.unkt else '<unk>'

    def get_pre(self, sub, j):
        return '<unkpre%d>'%j if self.precounts[sub] < self.unkt else sub

    def get_suf(self, sub,j):
        return  '<unksuf%d>'%j if self.sufcounts[sub] < self.unkt else sub

    def get_features(self, i, sent):
        ws = (['<s>']*self.n_context) + sent['ws'] + (['</s>']*self.n_context)
        wi = i + self.n_context
        w = ws[wi]
        c = self.n_context

        featd = {}

        if 'w' in self.feats:
            featd.update(('w%d'%j, self.get_w(w)) for w, j in zip(ws[wi-c:wi+c+1], range(-c,c+1)))
        """
        if 'o' in self.feats:
            featd.update(('cngram%dp%d'%(j,ni),ng) for j in range(2,5) for ni, ng in enumerate(self.f_ngrams(w,j)))
        """
        if 'p' in self.feats:
            # featd.update(('pre%d'%j, self.get_pre(w[:j])) for j in range(1,len(w)-1))
            featd.update(('pre%d'%j, self.get_pre(w[:j],j)) for j in range(2,5) if len(w)>j)
        if 's' in self.feats:
            # featd.update(('suf%d'%j, self.get_suf(w[:j])) for j in range(1,len(w)-1))
            featd.update(('suf%d'%j, self.get_suf(w[-j:],j)) for j in range(2,5) if len(w)>j)
        if 'n' in self.feats:
            featd.update(('cngram%d'%j, ng) for j in range(3,6) for ng in self.f_ngrams(w,j) if len(w)-2>=j)
        if 'c' in self.feats:
            featd['context'] =  ' '.join([self.get_w(w) for w in ws[wi-c:wi] + ['___'] + ws[wi+1:wi+c+1]])
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
    logging.debug(tabulate(confusion_matrix(np.array(gold_labels), np.array(pred_labels)), headers=[0,1]))
    p, r, f = evaluate_system.evaluateIdentifier(gold_labels, pred_labels)
    return p,r,f

# def main():
if __name__ == '__main__':
    random.seed(666)
    np.random.seed(1234567)

    parser = get_arg_parser()
    args = vars(parser.parse_args())

    setup_logger(args)

    logging.debug(tabulate([OrderedDict((k,v) for k,v in sorted(args.iteritems()))], headers='keys'))
    """
    for k,v in sorted(args.iteritems()):
        logging.critical('{} {}'.format(k,v))
    """


    if args['testf']:
        trn = utils.get_dset()
        tst = utils.get_test()
        if args['emb']:
            emb = Emb(trn+tst, args)
        ytrn, ytst = fit_predict(trn, tst, args)
        with open(args['testf'], 'w') as out:
            for y in ytrn: out.write('%d\n'%y)
    else:
        dset = utils.get_dset()
        random.shuffle(dset)
        if len(args['embs']):
            emb = Emb(dset, args)
        xvalidate(dset, args)

