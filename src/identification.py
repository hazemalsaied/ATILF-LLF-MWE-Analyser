import sys

import oracle
from classification import SVMClf
from corpus import *
from evaluation import evaluate
from extraction import extract
from parser import parse


def identify(outputPath, load=False, multipleFile=False):
    corpus = Corpus(multipleFile=multipleFile, load=load)
    if load:
        svm = SVMClf(None, None, load=load)
    else:
        oracle.parse(corpus)
        labels, data = extract(corpus)
        svm = SVMClf(labels, data, load=load, save=multipleFile)
    parse(corpus, svm.classifier, svm.verctorizer)
    with open(outputPath, 'w') as f:
        f.write(str(corpus))
    evaluate(corpus)


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
    with open(os.path.join(settings.PROJECT_PATH, '/Corpora/LPP/mweLEX.txt'), 'w') as F:
        F.write(res)
    return lexicon


reload(sys)
sys.setdefaultencoding('utf8')
logging.basicConfig(level=logging.WARNING)
identify('/Users/halsaied/Downloads/LPP-Simple/Corpora/AIW/mwe', load=False, multipleFile=True)
