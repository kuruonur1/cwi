from __future__ import division

if __name__ == '__main__':
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
