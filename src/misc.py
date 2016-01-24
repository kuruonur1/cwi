from __future__ import division
from collections import Counter
from utils import get_dset, get_test

def get_tagged_vocab(dset):
    return set(w for sent in dset for w,m in zip(sent['ws'],sent['ii']) if m)

def get_vocab(dset):
    return set(w for sent in dset for w in sent['ws'])

if __name__ == '__main__':
    trn = get_dset()
    tst = get_test()
    print map(len, map(get_tagged_vocab, [trn,tst]))
    print 'tagged vocab size trn {} tst {}'.format(*map(len, map(get_tagged_vocab, [trn,tst])))
    print 'vocab size trn {} tst {}'.format(*map(len, map(get_vocab, [trn,tst])))

    vtrn, vtst = map(get_tagged_vocab, [trn,tst])
    print 'vtst diff: {:.2f}'.format( len(vtst.difference(vtrn)) / len(vtst) )

    vtrn, vtst = map(get_vocab, [trn,tst])
    print 'vtst diff: {:.2f}'.format( len(vtst.difference(vtrn)) / len(vtst) )

    precnt = Counter(w[:j] for sent in trn for w, lbl in zip(sent['ws'],sent['ls']) for j in range(2,5) if lbl==1 and len(w)>j)
    sufcnt = Counter(w[-j:] for sent in trn for w, lbl in zip(sent['ws'],sent['ls']) for j in range(2,5) if lbl==1 and len(w)>j)
    print precnt.most_common(100)
    print sufcnt.most_common(100)

    """
    with open('data/cwi_training.txt') as src:
        lines = (l.strip().split('\t') for l in src)
        cw = set(l[1] for l in lines if l[-1] == '1')
        print len(cw)
        print list(cw)[:5]
    with open('/ai/home/vcirik/embeddings/wikipedia2MUNK-100.embeddings') as src:
        emb_words = set(l.strip().split(' ')[0] for l in src)
        print len(emb_words)
        print list(emb_words)[:5]
    print len(cw.difference(emb_words)) / len(cw)
    print cw.difference(emb_words)

    with open('data/cwi_testing.txt') as src:
        cw_test = set(l.strip().split('\t')[1] for l in src)
    print len(cw_test.difference(emb_words)) / len(cw_test)
    """
