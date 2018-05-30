import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import settings
from extraction import getFeatures
from transitions import *

mergeAsSet = {TransitionType.MERGE_AS_ID, TransitionType.MERGE_AS_IREFLV,
              TransitionType.MERGE_AS_VPC, TransitionType.MERGE_AS_LVC, TransitionType.MERGE_AS_OTH}


def parse(corpus, clf, vectorizer):
    initializeSent(corpus)
    printable = False
    for sent in corpus.testingSents:
        if printable:
            break
        printable = False
        if sent.text.lower().startswith('you must see to it that you regularly'):
            printable = True
            pass
        sent.initialTransition = Transition(None, isInitial=True, sent=sent)
        transition = sent.initialTransition
        while not transition.isTerminal():
            newTransition = nextTrans(transition, sent, clf, vectorizer, printable)
            newTransition.apply(transition, sent, parse=True, confidence=newTransition.confidence)
            transition = newTransition


def nextTrans(transition, sent, clf, vectorizer, printable=False):
    legalTansDic = transition.getLegalTransDic()
    featDic = getFeatures(transition, sent)
    if not isinstance(featDic, list):
        featDic = [featDic]
    transTypeValue = clf.predict(vectorizer.transform(featDic))[0]
    transType = getType(transTypeValue)
    if printable and False:
        printTransCoeff(featDic, clf, vectorizer)
    if transType in legalTansDic:
        trans = legalTansDic[transType]
        return trans
    if len(legalTansDic):
        return initialize(legalTansDic.keys()[0], sent, confidence=1)


def printTransCoeff(featDic, clf, vectorizer):
    shiftClsIdx = getTransClsIdx(clf, TransitionType.SHIFT.value)
    reduceClsIdx = getTransClsIdx(clf, TransitionType.REDUCE.value)
    whiteMergeClsIdx = getTransClsIdx(clf, TransitionType.WHITE_MERGE.value)
    mergeClsIdx = getTransClsIdx(clf, TransitionType.MERGE_AS_OTH.value)
    activeFeatDic = featDic[0]
    idxs = [mergeClsIdx, whiteMergeClsIdx, reduceClsIdx, shiftClsIdx]
    idxsLbls = ["merge", "whiteMerge", "reduce", "shift"]
    idxsLbl = 0

    total = 0.
    for f in activeFeatDic:
        completKey = f + '=' + str(activeFeatDic[f])
        if completKey in vectorizer.vocabulary_:
            featIdx = vectorizer.vocabulary_[completKey]
            for tranType in [0, 1]:  # idxs:
                # if tranType in clf.coef_ and featIdx in clf.coef_[tranType]:
                # total += abs(clf.coef_[tranType][featIdx])
                total += clf.coef_[tranType][featIdx]
    for csIdx in idxs:
        mainFeatDic = dict()
        for k in activeFeatDic:
            completKey = k + '=' + str(activeFeatDic[k])
            kParts = k.split('_')
            key = kParts[0]
            if len(kParts) > 1 and completKey in vectorizer.vocabulary_:
                featIdx = vectorizer.vocabulary_[completKey]
                val = clf.coef_[csIdx][featIdx]
                if val != 0:
                    if key not in mainFeatDic:
                        mainFeatDic[key] = val
                    else:
                        mainFeatDic[key] += val

        allFeatDic = dict()
        for k in activeFeatDic:
            completKey = k + '=' + str(activeFeatDic[k])
            if completKey in vectorizer.vocabulary_:
                featIdx = vectorizer.vocabulary_[completKey]
                allFeatDic[completKey] = clf.coef_[csIdx][featIdx]

        classTotal = 0.
        for k in activeFeatDic:
            completKey = k + '=' + str(activeFeatDic[k])
            if completKey in vectorizer.vocabulary_:
                featIdx = vectorizer.vocabulary_[completKey]
                classTotal += clf.coef_[csIdx][featIdx]
        print 'class coeff sum = ', classTotal
        for k in allFeatDic:
            # if int((allFeatDic[k] * 100 / total)):
            print k, ':', round((allFeatDic[k] * 10), 2)
        idxsLbl += 1


def getTransClsIdx(clf, transTypeValue):
    csIdx = 0
    for cs in clf.classes_:
        if cs == transTypeValue:
            break
        csIdx += 1
    return csIdx


def initializeSent(corpus):
    for sent in corpus.testingSents:
        sent.identifiedVMWEs = []
        sent.initialTransition = None


def plottt(a, b, labels, config, label=''):
    plotPath = os.path.join(settings.PROJECT_PATH, 'plots/')
    if not os.path.exists(plotPath):
        os.makedirs(plotPath)
    n = len(a)
    plt.figure()
    plt.scatter(a, b)
    for i in range(0, n):
        xy = (a[i], b[i])
        plt.annotate(labels[i], xy, rotation='vertical')
    red_patch = mpatches.Patch(color='red', label='Transition: ' + str(config))
    plt.legend(handles=[red_patch])
    if config.stack and len(config.stack) > 1:
        tokens1 = getTokens(config.stack[-1])
        tokens2 = getTokens(config.stack[-2])
        name = ''
        for tok in tokens2 + tokens1:
            name += tok.text + ' '
        plt.savefig(os.path.join(plotPath, name[:-1] + ('' if not label else '(Vs. ' + label + ')') + '.png'))
        plt.close()


def getValues(path, title=False):
    with open(path, 'r') as f:
        for line in f:
            if line:
                if line.split(':') > 1:
                    if title:
                        print (line.split(':')[0].strip())
                    else:
                        print float(line.split(':')[1].strip())


if __name__ == '__main__':
    getValues('/Users/halsaied/PycharmProjects/Cornell/ATILF-LLF-MWE-Analyser/Corpora/t.txt',
              False)
