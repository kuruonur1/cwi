import logging
from itertools import *
from tabulate import tabulate
from collections import defaultdict as dd

def setup_logger(args):
    import socket
    host = socket.gethostname().split('.')[0]
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    shandler = logging.StreamHandler()
    shandler.setLevel(logging.CRITICAL)
    logger.addHandler(shandler);

    if len(args['log']) > 0 and args['log'] != 'nothing':
        ihandler = logging.FileHandler('logs/{}.log'.format(args['log']), mode='w')
        ihandler.setLevel(logging.DEBUG)
        logger.addHandler(ihandler);

def get_dset():
    s2tuples = dd(list)
    with open('data/cwi_training.txt') as src:
        lines = [l.strip().split('\t') for l in src]
    for s,w,i,l in lines:
        s2tuples[s].append((int(i),w,int(l)))

    for tuples in s2tuples.itervalues():
        tuples.sort()

    trn = [st2sent(s,tuples) for s, tuples in s2tuples.items()]
    for sent in trn:
        sent['cseq'] = [c for w in sent['ws'] for c in chain(['\w'],w,['w/'])]
        sent['wiseq'] = [wi for wi,w in enumerate(sent['ws']) for c in chain(['\w'],w,['w/'])]
        sent['lseq'] = [l for w,l in zip(sent['ws'],sent['ls']) for c in chain(['\w'],w,['w/'])]
        sent['eseq'] = [int(c=='w/') for w in sent['ws'] for e, c in enumerate(chain(['\w'],w,['w/']))]
    return trn

def st2sent(s,tuples):
    sent = {}
    sent['ws'] = s.split(' ')
    sent['ls'] = [0]*len(sent['ws'])
    sent['ii'] = [0]*len(sent['ws'])
    for t in tuples:
        sent['ii'][t[0]] = 1
        sent['ls'][t[0]] = t[2]
    return sent

def pprint_word(sent):
    print tabulate([sent['ws'],sent['ls'],sent['ii']])

def pprint_char(sent):
    print tabulate([sent['cseq'][:40],sent['wiseq'][:40],sent['lseq'][:40],sent['eseq'][:40]])

def stats(s2tuples):
    print '----'
    print 'perc of tagged words per sent:',  np.mean([len(tuples)/len(s.split(' ')) for s, tuples in s2tuples.items()])

    cw = set(t[2] for tuples in s2tuples.itervalues() for t in tuples if t[2] == 1)
    ncw = set(t[2] for tuples in s2tuples.itervalues() for t in tuples if t[2] == 0)
    print len(cw.intersection(ncw))/len(cw)
    print len(cw.intersection(ncw))/len(ncw)

    samp = random.sample(s2tuples.items(),10)
    for s, tuples in samp:
        print s
        print tuples
        print
    print '-----'
    print

if __name__ == '__main__':
    import random
    trn = get_dset()
    sent = random.choice(trn)
    pprint_word(sent)
    pprint_char(sent)
