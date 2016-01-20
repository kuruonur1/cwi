import logging
from itertools import *
from tabulate import tabulate
from collections import defaultdict as dd


def get_dset():
    s2tuples = dd(list)
    with open('data/cwi_training.txt') as src:
        lines = [l.strip().split('\t') for l in src]
    for s,w,i,l in lines:
        s2tuples[s].append((int(i),w,int(l)))

    for tuples in s2tuples.itervalues():
        tuples.sort()

    trn = [st2sent(s,tuples) for s, tuples in s2tuples.items()]
    return trn

def get_test():
    s2tuples = dd(list)
    with open('data/cwi_testing.txt') as src:
        lines = [l.strip().split('\t') for l in src]

    prevs = 'asdf'
    tuples = []
    tst = []
    for curs,w,i in lines:
        if prevs != curs and len(tuples) > 0:
            tst.append(st2sent(prevs, tuples))
            tuples = []
        tuples.append((int(i),w,0))
        prevs = curs
    tst.append(st2sent(prevs, tuples))
    return tst


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
    tst = get_test()
    print len(tst)
    print 'trn len:', sum(sum(sent['ii']) for sent in trn)
    print 'tst len:', sum(sum(sent['ii']) for sent in tst)
    print 'tst len:', len([w for sent in tst for m, w in zip(sent['ii'], sent['ws']) if m])
    pprint_word(tst[0])



