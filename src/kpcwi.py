from __future__ import division

import random
import logging
import argparse
import numpy as np

from itertools import *
from tabulate import tabulate
from collections import Counter

import utils
import evaluate_system

from kperceptron import KPerceptron

def get_arg_parser():
    parser = argparse.ArgumentParser(prog="cwi")
    
    parser.add_argument("--emb", default='mikolovWikipedia100.embeddings', help="how to merge forward backward layer outputs")
    parser.add_argument("--n_batch", default=100, type=int, help="batch size")
    parser.add_argument("--n_fold", default=5, type=int, help="batch size")
    parser.add_argument("--n_context", default=2, type=int, help="num of words around token")
    parser.add_argument("--fepoch", default=100, type=int, help="number of epochs")
    parser.add_argument("--patience", default=10, type=int, help="angerrr")
    parser.add_argument("--log", default='nothing', help="log file name")


    parser.add_argument("--clf", default='per', help="clf type")
    parser.add_argument("--kerntype", default='poly', choices=['lin','poly','rbf'],  help="kernel type")
    parser.add_argument("--kerngamma", default=1., type=float, help="poly or rbf gamma")
    parser.add_argument("--kerncoef0", default=1., type=float, help="poly kernel coef0")
    parser.add_argument("--kerndegree", default=2, type=int, help="poly kernel degree")

    parser.add_argument("--C", default=1., type=float, help="C for SVC")

    return parser

def xvalidate(dset, args):
    vocab = set(w for sent in dset for w in sent['ws'])
    emb = EmbRand() if args['emb'] == 'rand' else Emb(args['emb'], vocab)
    feat = Feat(emb)
    from sklearn.feature_extraction import DictVectorizer
    dvec = DictVectorizer(sparse=False)
    dvec.fit(feat.get_features(i, sent) for sent in dset for i,w in enumerate(sent['ws']))
    # logging.debug(dvec.feature_names_)

    fold_indxs = [0,40] if args['n_fold'] == 1 else range(0,len(dset),int(np.ceil(len(dset)/args['n_fold']))) + [len(dset)]
    fold_scores = []
    for fstart, fend in zip(fold_indxs, fold_indxs[1:]):
        logging.debug('fstart:{} fend:{}'.format(fstart,fend))
        trn = dset[:fstart] + dset[fend:]
        dev = dset[fstart:fend]

        f = validate(trn, dev, dvec, feat, args)
        fold_scores.append(f)
    f1mean, f1std = np.mean(fold_scores), np.std(fold_scores)
    logging.debug('mean:{:.2f} std:{:.2f}'.format(f1mean, f1std))
    return f1mean, f1std

def validate(trn, dev, dvec, feat, args):
    Xtrn = dvec.transform(feat.get_features(i, sent) for sent in trn for m,(i,w) in zip(sent['ii'],enumerate(sent['ws'])) if m)
    Xdev = dvec.transform(feat.get_features(i, sent) for sent in dev for m,(i,w) in zip(sent['ii'],enumerate(sent['ws'])) if m)
    ytrn = np.array([lbl for sent in trn for m,lbl in zip(sent['ii'],sent['ls']) if m])
    ydev = np.array([lbl for sent in dev for m,lbl in zip(sent['ii'],sent['ls']) if m])
    cweights = {0:(ytrn==1).sum()/ytrn.size, 1:(ytrn==0).sum()/ytrn.size}

    logging.debug('Xtrn shape:{}'.format(Xtrn.shape))

    from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif

    sel = SelectKBest(f_classif, k=3000)
    # sel = VarianceThreshold(threshold=.01)
    sel.fit(np.vstack((Xtrn,Xdev)), np.concatenate((ytrn,ydev)))
    Xtrn, Xdev = sel.transform(Xtrn), sel.transform(Xdev)
    # logging.debug(sel.scores_)
    logging.debug('after sel: Xtrn:{} Xdev: {}'.format(Xtrn.shape, Xdev.shape))
    logging.debug([(fea, score) for score, fea in islice(reversed(sorted(zip(sel.scores_, dvec.feature_names_))), 500)])


    # logging.debug('norm**2 mean:{}'.format(np.mean(np.linalg.norm(Xtrn-Xtrn[10],axis=1)**2)))

    from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
    from sklearn.svm import SVC, LinearSVC
    from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier, Perceptron, LogisticRegression

    if args['clf'] == 'svm' or args['clf'] == 'lo':
        if args['clf'] == 'lo':
            clf = LogisticRegression(class_weight=cweights)
        if args['kerntype'] == 'lin':
            clf = LinearSVC(C=args['C'], class_weight=cweights)
        else:
            clf = SVC(class_weight=cweights, C=args['C'], kernel=args['kerntype'], gamma=args['kerngamma'], degree=args['kerndegree'])
        clf.fit(Xtrn, ytrn)

        yhat, acctrn = clf.predict(Xtrn), clf.score(Xtrn, ytrn)
        row1 = list(evalu(trn, yhat)) # p, r, f

        yhat, accdev = clf.predict(Xdev), clf.score(Xdev, ydev)
        row2 = list(evalu(dev, yhat)) # p, r, f
        rows = [['trn', acctrn] + row1, ['dev', accdev] + row2]
        logging.debug(tabulate(rows, floatfmt='.2f', headers=['dset','wacc','p','r','f']))

        max_f1=row2[-1]
    else:
        if args['clf'] == 'nb':
            clf = GaussianNB()
        elif args['clf'] == 'kper':
            clf = KPerceptron(kerntype=args['kerntype'],kerngamma=args['kerngamma'],
                    kerndegree=args['kerndegree'], kerncoef0=args['kerncoef0'], ybias=1)
        elif args['clf'] == 'per':
            logging.debug(cweights)
            clf = Perceptron(class_weight=cweights, eta0=.1)
        elif args['clf'] == 'sgd':
            clf = SGDClassifier(penalty='elasticnet',class_weight=cweights)
        elif args['clf'] == 'pac':
            clf = PassiveAggressiveClassifier(C=args['C'], loss='hinge')
        else:
            raise Exception()

        f1s = []
        for e in range(1,args['fepoch']+1):
            clf.partial_fit(Xtrn, ytrn, [0,1])
            # clf.fit(Xtrn, ytrn)

            yhat, acctrn = clf.predict(Xtrn), clf.score(Xtrn, ytrn)
            row1 = list(evalu(trn, yhat)) # p, r, f

            yhat, accdev = clf.predict(Xdev), clf.score(Xdev, ydev)
            row2 = list(evalu(dev, yhat)) # p, r, f
            f1s.append((row2[-1],e))
            rows = [['trn',e, acctrn] + row1, ['dev',e, accdev] + row2]
            logging.debug(tabulate(rows, floatfmt='.2f', headers=['dset','epoch','wacc','p','r','f']))

            if e - max(f1s)[1] > args['patience']: break
        max_f1 = max(f1s)[0]
    return max_f1

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

class Emb(object):
    
    def __init__(self, fname, vocab):
        # with open('/ai/home/vcirik/embeddings/wikipedia2MUNK-25.embeddings') as src:
        fname = '/ai/home/vcirik/embeddings/{}'.format(fname)
        logging.debug(fname)
        with open(fname) as src:
            srcgen = (l.strip().split(' ') for l in src)
            self.d = dict((l[0], map(float,l[1:])) for l in srcgen if l[0] in vocab or l[0] == '*UNKNOWN*')

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

    def get_token(self, sent):
        Xsent = np.array([self.get_w(w) for w in sent['ws']])
        return Xsent


    def get_context(self, sent, c=2):
        ws_padded = (['<s>']*c) + sent['ws'] + (['</s>']*c)
        # logging.debug(ws_padded)
        Xsent = np.array([[fi for w in ws_padded[(i-c):(i+c+1)] for fi in self.get_w(w)] for i in range(c,c+len(sent['ws']))])
        return Xsent

class Feat(object):

    def __init__(self, emb):
        self.emb = emb

    def get_features(self, i, sent, c=2):
        featd = {}
        w = sent['ws'][i]

        ws_padded = (['<s>']*c) + sent['ws'] + (['</s>']*c)
        # featd = dict(('e%d'%fi, fv) for fi, fv in enumerate(fv for w in ws_padded[(i-c):(i+c+1)] for fv in self.emb.get_w(w)))

        featd.update(('substr%d'%j, w[:j]) for j in range(1,len(w)))
        # featd.update(('w%d'%j, ws_padded[i+j]) for j in range(-c,c+1))
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

            

def main():
    random.seed(0)
    np.random.seed(1234567)

    parser = get_arg_parser()
    args = vars(parser.parse_args())

    setup_logger(args)

    for k,v in sorted(args.iteritems()):
        logging.critical('{} {}'.format(k,v))

    dset = utils.get_dset()
    sent_lens = [len(sent['ws']) for sent in dset]
    logging.debug('# of words per sent, min:{} max:{} mean:{:.2f} std:{:.2f}'.format(min(sent_lens), max(sent_lens), np.mean(sent_lens), np.std(sent_lens)))
    sent = dset[0]
    utils.pprint_word(sent)

    xvalidate(dset, args)

if __name__ == '__main__':
    main()
