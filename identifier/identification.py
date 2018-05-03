import sys

import oracle
from classification import SVMClf
from corpus import *
from evaluation import evaluate
from extraction import extract
from parser import parse


def identify(outputPath, load=False, multipleFile=False):
    corpus = Corpus(multipleFile=multipleFile, load=load)
    print 'Corpus loaded'
    if load:
        svm = SVMClf(None, None, load=load)
    else:
        oracle.parse(corpus)
        print 'parse finished'
        labels, data = extract(corpus)
        print 'Training data was prepared'
        svm = SVMClf(labels, data, load=load, save=multipleFile)
    print 'Evaluation started'
    parse(corpus, svm.classifier, svm.verctorizer)
    with open(outputPath, 'w') as f:
        f.write(str(corpus))
    evaluate(corpus)
    print 'finished'


def getTrainLexicon():
    corpus = Corpus(multipleFile=False)
    lexicon = {}
    for sent in corpus.trainingSents:
        for mwe in sent.vMWEs:
            if len(mwe.tokens) > 1:
                lexicon[getTokenText(mwe.tokens)] = True
    res = ''
    for k in sorted(lexicon.keys()):
        res += k + '\n'
    with open(os.path.join(settings.PROJECT_PATH, '/LPP/mweLEX.txt'), 'w') as F:
        F.write(res)
    return lexicon


reload(sys)
sys.setdefaultencoding('utf8')
logging.basicConfig(level=logging.WARNING)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))[:-3]
print ROOT_DIR
identify(os.path.join(ROOT_DIR, 'Corpora/AIW/mwe.auto'), load=True, multipleFile=True)
