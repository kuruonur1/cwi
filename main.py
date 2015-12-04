from __future__ import division

import numpy as np
import random

from collections import defaultdict as dd
from tabulate import tabulate

def main():
    s2tuples = dd(list)
    with open('cwi_training.txt') as src:
        lines = [l.strip().split('\t') for l in src]
    for s,w,i,l in lines:
        s2tuples[s].append((int(i),w,int(l)))

    for tuples in s2tuples.itervalues():
        tuples.sort()

    trn = [st2sent(s,tuples) for s, tuples in s2tuples.items()]
    sent = random.choice(trn)
    print tabulate([sent['ws'],sent['ts']], tablefmt='plain')

def st2sent(s,tuples):
    sent = {}
    sent['ws'] = s.split(' ')
    sent['ts'] = [0]*len(sent['ws'])
    for t in tuples:
        if t[2] == 1:
            sent['ts'][t[0]] = 1
    return sent
    # stats(s2tuples)


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
    main()
