import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import settings
from extraction import getFeatures
from transitions import *

mergeAsSet = {TransitionType.MERGE_AS_ID, TransitionType.MERGE_AS_IREFLV,
              TransitionType.MERGE_AS_VPC, TransitionType.MERGE_AS_LVC, TransitionType.MERGE_AS_OTH}


def parse(corpus, clf, vectorizer):
    initializeSent(corpus)
    for sent in corpus.testingSents:
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
    if len(legalTansDic) == 1:
        return initialize(legalTansDic.keys()[0], sent, confidence=1)

    featDic = getFeatures(transition, sent)
    if not isinstance(featDic, list):
        featDic = [featDic]
    probabilities = clf.predict_proba(vectorizer.transform(featDic))[0]
    confidence = max(probabilities)
    transTypeValue = clf.predict(vectorizer.transform(featDic))[0]
    transType = getType(transTypeValue)
    if printable:
        printTransCoeff(transition, transTypeValue, featDic, clf, vectorizer)
    if transType in legalTansDic:
        trans = legalTansDic[transType]
        trans.confidence = confidence
        return trans
    if len(legalTansDic):
        return initialize(legalTansDic.keys()[0], sent, confidence=1)


def printTransCoeff(transition, transTypeValue, featDic, clf, vectorizer):
    print transition
    # choosenClsIdx = getTransClsIdx(clf, transTypeValue)
    shiftClsIdx = getTransClsIdx(clf, TransitionType.SHIFT.value)
    reduceClsIdx = getTransClsIdx(clf, TransitionType.REDUCE.value)
    whiteMergeClsIdx = getTransClsIdx(clf, TransitionType.WHITE_MERGE.value)
    mergeClsIdx = getTransClsIdx(clf, TransitionType.MERGE_AS_OTH.value)
    activeFeatDic = featDic[0]
    idxs = [mergeClsIdx, whiteMergeClsIdx, reduceClsIdx, shiftClsIdx]
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
        absValues = np.absolute(mainFeatDic.values())
        totalCoeffecient = sum(absValues)
        allDeatDic = dict()
        for k in activeFeatDic:
            completKey = k + '=' + str(activeFeatDic[k])
            if completKey in vectorizer.vocabulary_:
                featIdx = vectorizer.vocabulary_[completKey]
                allDeatDic[completKey] = clf.coef_[csIdx][featIdx]
        absValues = np.absolute(allDeatDic.values())
        total = sum(absValues)
        # print "#" * 20 + '\n' + str(csIdx) + '\n' + "#" * 20
        # for k in mainFeatDic:
        #    print k, ':', int((mainFeatDic[k] * 100 / totalCoeffecient))
        print "#" * 20 + '\n' + str(csIdx) + '\n' + "#" * 20
        for k in allDeatDic:
            if int((allDeatDic[k] * 100 / total)):
                print k, ':', int((allDeatDic[k] * 100 / total))


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
                        print int(line.split(':')[1].strip())



if __name__ == '__main__':
    getValues('/Users/halsaied/PycharmProjects/Cornell/ATILF-LLF-MWE-Analyser/Corpora/t.txt', True)
