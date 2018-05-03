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
    for sent in corpus.testingSents:
        sent.initialTransition = Transition(None, isInitial=True, sent=sent)
        transition = sent.initialTransition
        while not transition.isTerminal():
            newTransition = nextTrans(transition, sent, clf, vectorizer)
            newTransition.apply(transition, sent, parse=True, confidence=newTransition.confidence)
            transition = newTransition


def nextTrans(transition, sent, clf, vectorizer):
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
    if transType in legalTansDic:
        if transType in mergeAsSet:
            csIdx = 0
            for cs in clf.classes_:
                if cs == transTypeValue:
                    break
                csIdx += 1
            cs1Idx = 0
            for cs in clf.classes_:
                if cs == TransitionType.SHIFT.value:
                    break
                cs1Idx += 1
            cs2Idx = 0
            for cs in clf.classes_:
                if cs == TransitionType.REDUCE.value:
                    break
                cs2Idx += 1
            activeFeatDic = featDic[0]

            mainFeatDic = dict()
            for k in activeFeatDic:
                completKey = k + '=' + str(activeFeatDic[k])
                kParts = k.split('_')
                key = kParts[0]
                if len(kParts) > 1 and completKey in vectorizer.vocabulary_:
                    featIdx = vectorizer.vocabulary_[completKey]
                    val = clf.coef_[csIdx][featIdx]
                    if val > 0:
                        if key not in mainFeatDic:
                            mainFeatDic[key] = val
                        else:
                            mainFeatDic[key] += val
            totalCoeffecient = .0
            for v in mainFeatDic.values():
                totalCoeffecient += v
            for k in mainFeatDic:
                mainFeatDic[k] = int((mainFeatDic[k] / totalCoeffecient) * 100)
            if 'L' not in mainFeatDic:
                print activeFeatDic
                print sent
                for k in mainFeatDic:
                    print k, ',', mainFeatDic[k]
        trans = legalTansDic[transType]
        trans.confidence = confidence
        return trans
    if len(legalTansDic):
        return initialize(legalTansDic.keys()[0], sent, confidence=1)


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
