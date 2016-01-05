import string, numpy as np, logging
from itertools import *

from sklearn.feature_extraction import DictVectorizer

class Feat(object):

    def __init__(self):
        self.dvec = DictVectorizer(dtype=np.float32, sparse=False)

    def fit(self, trn):
    # def fit(self, trn, dev, tst):
        self.dvec.fit(self.feat_basic(ci, sent)  for sent in trn for ci,c in enumerate(sent['cseq']))
        # self.tseqenc.fit([t for sent in trn for t in sent['tseq']])
        # self.tsenc.fit([t for sent in chain(trn,dev,tst) for t in sent['ts']])
        self.feature_names = self.dvec.get_feature_names()
        # self.ctag_classes = self.tseqenc.classes_
        # self.wtag_classes = self.tsenc.classes_
        logging.info(self.feature_names)
        logging.debug(' '.join([fn for fn in self.feature_names]))
        # logging.info(self.ctag_classes)
        # logging.info(self.wtag_classes)
        self.NF = len(self.feature_names)
        # logging.info('NF: {} NC: {}'.format(self.NF, self.NC))
        logging.info('NF: {}'.format(self.NF))

    def transform(self, sent):
        Xsent = self.dvec.transform([self.feat_basic(ci, sent) for ci,c in enumerate(sent['cseq'])]) # nchar x nf
        slen = Xsent.shape[0]
        ysent = np.zeros((slen, 2), dtype=bool)
        ysent[range(slen), sent['lseq']] = True
        # ysent = np.array(sent['lseq'])
        return Xsent, ysent

    def feat_basic(self, ci, sent):
        return {'c': sent['cseq'][ci]}

        
