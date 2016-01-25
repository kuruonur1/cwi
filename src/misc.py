from __future__ import division
from collections import Counter
from utils import get_dset, get_test, pprint_word

def get_tagged_vocab(dset):
    return set(w for sent in dset for w,m in zip(sent['ws'],sent['ii']) if m)

def get_vocab(dset):
    return set(w for sent in dset for w in sent['ws'])

def get_contexts(sent, c):
    ws = (['<s>']*c) + sent['ws'] + (['</s>']*c)

    contexts = []
    for i, w in enumerate(sent['ws']):
        wi = i + c
        if sent['ii'][i]:
            contexts.append(' '.join([w for w in ws[wi-c:wi] + ['___'] + ws[wi+1:wi+c+1]]))
    return contexts

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

    precnt = Counter(w[:j] for sent in trn for w, lbl in zip(sent['ws'],sent['ls']) for j in range(3,5) if lbl==1 and len(w)>j)
    sufcnt = Counter(w[-j:] for sent in trn for w, lbl in zip(sent['ws'],sent['ls']) for j in range(3,5) if lbl==1 and len(w)>j)
    print 'most common prefixes:', precnt.most_common(100)
    print 'most common suffixes:', sufcnt.most_common(100)

    trn_tagged_wcounts = Counter(w for sent in trn for w, lbl, m in zip(sent['ws'],sent['ls'],sent['ii']) if m)
    print 'perc of words appers 1+ in trn:', sum(c for w,c in trn_tagged_wcounts.iteritems() if c > 1) / sum(c for w,c in trn_tagged_wcounts.iteritems())

    tst_tagged_wcounts = Counter(w for sent in tst for w, lbl, m in zip(sent['ws'],sent['ls'],sent['ii']) if m)
    print 'most common tst_tagged_wcounts:', tst_tagged_wcounts.most_common(100)
    print 'perc of words appers 1+ in tst:', sum(c for w,c in tst_tagged_wcounts.iteritems() if c > 1) / sum(c for w,c in tst_tagged_wcounts.iteritems())

    context_counts = Counter(context for sent in trn for context in get_contexts(sent, 1))
    print 'most common contexts in trn:', context_counts.most_common(100)

    context_counts = Counter(context for sent in tst for context in get_contexts(sent, 1))
    print 'most common contexts in tst:', context_counts.most_common(100)
