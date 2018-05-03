from corpus import *
from param import FeatParams


def extract(corpus):
    labels, featureDic = [], []
    for sent in corpus.trainingSents:
        labelsTmp, featuresTmp = extractSent(sent)
        labels.extend(labelsTmp)
        featureDic.extend(featuresTmp)
    return labels, featureDic


def extractSent(sent):
    transition = sent.initialTransition
    labels, features = [], []
    while transition.next:
        if transition.next and transition.next.type:
            labels.append(transition.next.type.value)
            feats = getFeatures(transition, sent)
            features.append(feats)
            transition = transition.next
    sent.featuresInfo = [labels, features]
    return labels, features


def getFeatures(transition, sent):
    transDic = {}
    configuration = transition.configuration
    # TODO return transDic directly in this case
    if FeatParams.useStackLength and len(configuration.stack) > 1:
        transDic['O_StackLengthIs'] = len(configuration.stack)

    if len(configuration.stack) >= 2:
        stackElements = [configuration.stack[-2], configuration.stack[-1]]
    else:
        stackElements = configuration.stack

    # General linguistic Informations
    if stackElements:
        elemIdx = len(stackElements) - 1
        for elem in stackElements:
            generateLinguisticFeatures(elem, 'A_S' + str(elemIdx), transDic)
            elemIdx -= 1

    if len(configuration.buffer) > 0:
        if FeatParams.useFirstBufferElement:
            generateLinguisticFeatures(configuration.buffer[0], 'A_B0', transDic)

        if FeatParams.useSecondBufferElement and len(configuration.buffer) > 1:
            generateLinguisticFeatures(configuration.buffer[1], 'E_B1', transDic)

    # Bi-Gram Generation
    if FeatParams.useBiGram:
        if len(stackElements) > 1:
            # Generate a Bi-gram S1S0 S0B0 S1B0 S0B1
            generateBiGram(stackElements[-2], stackElements[-1], 'F_S1 S0', transDic)
            if FeatParams.generateS1B1 and len(configuration.buffer) > 1:
                generateBiGram(stackElements[-2], configuration.buffer[1], 'F_S1 B1', transDic)
        if len(stackElements) > 0 and len(configuration.buffer) > 0:
            generateBiGram(stackElements[-1], configuration.buffer[0], 'F_S0 B0', transDic)
            if len(stackElements) > 1:
                generateBiGram(stackElements[-2], configuration.buffer[0], 'F_S1 B0', transDic)
            if len(configuration.buffer) > 1:
                generateBiGram(stackElements[-1], configuration.buffer[1], 'F_S0 B1', transDic)
                if FeatParams.generateS0B2Bigram and len(configuration.buffer) > 2:
                    generateBiGram(stackElements[-1], configuration.buffer[2], 'F_S0 B2', transDic)

    # Tri-Gram Generation
    if FeatParams.useTriGram and len(stackElements) > 1 and len(configuration.buffer) > 0:
        generateTriGram(stackElements[-2], stackElements[-1], configuration.buffer[0], 'I_S1 S0 B0', transDic)

    # Syntaxic Informations
    if len(stackElements) > 0 and FeatParams.useSyntax:
        generateSyntaxicFeatures(configuration.stack, configuration.buffer, transDic)

    # Distance information
    if FeatParams.useS0B0Distance and len(configuration.stack) > 0 and len(configuration.buffer) > 0:
        stackTokens = getTokens(configuration.stack[-1])
        transDic['M_S0 B0 Distance'] = str(
            sent.tokens.index(configuration.buffer[0]) - sent.tokens.index(stackTokens[-1]))
    if FeatParams.useS0S1Distance and len(configuration.stack) > 1 and isinstance(configuration.stack[-1], Token) \
            and isinstance(configuration.stack[-2], Token):
        transDic['K_S0 S1 Distance'] = str(
            sent.tokens.index(configuration.stack[-1]) - sent.tokens.index(configuration.stack[-2]))
    addTransitionHistory(transition, transDic)

    if FeatParams.useLexic and len(configuration.buffer) > 0 and len(configuration.stack) >= 1:
        generateDisconinousFeatures(configuration, sent, transDic)

    enhanceMerge(transition, transDic)

    return transDic


def enhanceMerge(transition, transDic):
    if not FeatParams.enhanceMerge:
        return
    config = transition.configuration
    if transition.type.value != 0 and len(config.buffer) > 0 and len(
            config.stack) > 0 and isinstance(config.stack[-1], Token):
        if isinstance(config.stack[-1], Token) and areInLexic([config.stack[-1], config.buffer[0]]):
            transDic['P_In Lexicon( S0, B0)'] = True

        if len(config.buffer) > 1 and areInLexic([config.stack[-1], config.buffer[0], config.buffer[1]]):
            transDic['P_In Lexicon( S0, B0, B1 )'] = True
        if len(config.buffer) > 2 and areInLexic(
                [config.stack[-1], config.buffer[0], config.buffer[1], config.buffer[2]]):
            transDic['P_In Lexicon( S0, B0, B1, B2)'] = True
        if len(config.buffer) > 1 and len(config.stack) > 1 and areInLexic(
                [config.stack[-2], config.stack[-1], config.buffer[1]]):
            transDic['P_In Lexicon( S1, S0, B1 )'] = True

    if len(config.buffer) > 0 and len(config.stack) > 1 and areInLexic(
            [config.stack[-2], config.buffer[0]]) and not areInLexic(
        [config.stack[-1], config.buffer[0]]):
        transDic['P_In Lexicon( S1, B0)'] = True
        transDic['P_In Lexicon( S0, B0)'] = False
        if len(config.buffer) > 1 and areInLexic(
                [config.stack[-2], config.buffer[1]]) and not areInLexic(
            [config.stack[-1], config.buffer[1]]):
            transDic['P_In Lexicon( S1, B1)'] = True
            transDic['P_In Lexicon( S0, B1)'] = False


def generateDisconinousFeatures(configuration, sent, transDic):
    tokens = getTokens([configuration.stack[-1]])
    tokenTxt = getTokenLemmas(tokens)
    for key in Corpus.mweDictionary.keys():
        if tokenTxt in key and tokenTxt != key:
            bufidx = 0
            for bufElem in configuration.buffer[:5]:
                if bufElem.lemma != '' and (
                        (tokenTxt + ' ' + bufElem.lemma) in key or (bufElem.lemma + ' ' + tokenTxt) in key):
                    transDic['L_S0 B' + str(bufidx) + ' Parts Of MWE'] = True
                    transDic['L_S0 B' + str(bufidx) + 'Parts Of MWE Distance'] = sent.tokens.index(
                        bufElem) - sent.tokens.index(tokens[-1])
                bufidx += 1
            break


def generateLinguisticFeatures(token, label, transDic):
    if isinstance(token, list):
        token = concatenateTokens([token])[0]
    transDic[label + ' Token'] = token.text
    if FeatParams.usePOS and token.posTag is not None and token.posTag.strip() != '':
        transDic[label + ' POS'] = token.posTag
    if FeatParams.useLemma and token.lemma is not None and token.lemma.strip() != '':
        transDic[label + ' Lemma'] = token.lemma
    if not FeatParams.useLemma and not FeatParams.usePOS:
        transDic[label + 'Last Three Letters'] = token.text[-3:]
        transDic[label + 'Last Two Letters'] = token.text[-2:]
    if FeatParams.useDictionary and ((token.lemma != '' and token.lemma in Corpus.mweTokenDic.keys())
                                     or token.text in Corpus.mweTokenDic.keys()):
        transDic['L_' + label[2:] + ' In Lexicon'] = 'true'


def generateSyntaxicFeatures(stack, buffer, dic):
    if stack and isinstance(stack[-1], Token):
        stack0 = stack[-1]
        if int(stack0.dependencyParent) == -1 or int(
                stack0.dependencyParent) == 0 or stack0.dependencyLabel.strip() == '' or not buffer:
            return
        for bElem in buffer:
            if bElem.dependencyParent == stack0.position:
                dic['H_Righ Dep ' + bElem.dependencyLabel] = 'true'
                dic['H_' + stack0.getLemma() + 'Righ Dep ' + bElem.dependencyLabel] = 'true'
                dic['H_' + stack0.getLemma() + ' ' + bElem.getLemma() + ' Righ Dep' + bElem.dependencyLabel] = 'true'

        if stack0.dependencyParent > stack0.position:
            for bElem in buffer:
                if bElem.position == stack0.dependencyParent:
                    dic['H_' + stack0.lemma + ' Gouverned By ' + bElem.getLemma()] = 'true'
                    dic[
                        'H_' + stack0.lemma + ' Gouverned By ' + bElem.getLemma() + ' ' + stack0.dependencyLabel] = 'true'
                    break
        if len(stack) > 1:
            stack1 = stack[-2]
            if not isinstance(stack1, Token):
                return
            if stack0.dependencyParent == stack1.position:
                dic['H_Syntaxic Relation ' + '+ ' + stack0.dependencyLabel] = True
            elif stack0.position == stack1.dependencyParent:
                dic['H_Syntaxic Relation' + '- ' + stack1.dependencyLabel] = True


def generateTriGram(token0, token1, token2, label, transDic):
    tokens = concatenateTokens([token0, token1, token2])
    getFeatureInfo(transDic, label + 'TTT ', tokens, 'ttt')
    getFeatureInfo(transDic, label + 'LLL ', tokens, 'lll')
    getFeatureInfo(transDic, label + 'PPP ', tokens, 'ppp')
    getFeatureInfo(transDic, label + 'LPP ', tokens, 'lpp')
    getFeatureInfo(transDic, label + 'PLP ', tokens, 'plp')
    getFeatureInfo(transDic, label + 'PPL ', tokens, 'ppl')
    getFeatureInfo(transDic, label + 'LLP ', tokens, 'llp')
    getFeatureInfo(transDic, label + 'LPL ', tokens, 'lpl')
    getFeatureInfo(transDic, label + 'PLL ', tokens, 'pll')


def generateBiGram(token0, token1, label, transDic):
    tokens = concatenateTokens([token0, token1])
    getFeatureInfo(transDic, label + 'TT ', tokens, 'tt')
    getFeatureInfo(transDic, label + 'LL ', tokens, 'll')
    getFeatureInfo(transDic, label + 'PP ', tokens, 'pp')
    getFeatureInfo(transDic, label + 'LP ', tokens, 'lp')
    getFeatureInfo(transDic, label + 'PL ', tokens, 'pl')


def concatenateTokens(tokens):
    idx = 0
    tokenDic = {}
    result = []
    for token in tokens:
        if isinstance(token, Token):
            result.append(Token(-1, token.text, token.lemma, token.posTag))
        elif isinstance(token, list):
            tokenDic[idx] = Token(-1, '', '', '')
            for subToken in getTokens(token):
                tokenDic[idx].text += subToken.text + '_'
                tokenDic[idx].lemma += subToken.lemma + '_'
                tokenDic[idx].posTag += subToken.posTag + '_'
            tokenDic[idx].text = tokenDic[idx].text[:-1]
            tokenDic[idx].lemma = tokenDic[idx].lemma[:-1]
            tokenDic[idx].posTag = tokenDic[idx].posTag[:-1]
            result.append(tokenDic[idx])
        idx += 1
    return result


def getFeatureInfo(dic, label, tokens, features):
    feature = ''
    idx = 0
    for token in tokens:
        if features[idx].lower() == 'l':
            if FeatParams.useLemma:
                if token.lemma.strip() != '':
                    feature += token.lemma.strip() + '_'
                else:
                    feature += '*' + '_'
        elif features[idx].lower() == 'p':
            if FeatParams.usePOS:
                if token.posTag.strip() != '':
                    feature += token.posTag.strip() + '_'
                else:
                    feature += '*' + '_'
        elif features[idx].lower() == 't':
            if token.text.strip() != '':
                feature += token.text.strip() + '_'
        idx += 1
    if len(feature) > 0:
        feature = feature[:-1]
        dic[label] = feature

    return ''


def areInLexic(tokensList):
    if getTokenLemmas(tokensList) in Corpus.mweDictionary.keys():
        return True
    return False


def addTransitionHistory(transition, transDic):
    if FeatParams.historyLength1:
        getTransitionHistory(transition, 1, 'B_Trans History (1)', transDic)
    if FeatParams.historyLength2:
        getTransitionHistory(transition, 2, 'B_Trans History (2)', transDic)
    if FeatParams.historyLength3:
        getTransitionHistory(transition, 3, 'B_Trans History (3)', transDic)


def getTransitionHistory(transition, length, label, transDic):
    idx = 0
    history = ''
    transRef = transition
    transition = transition.previous
    while transition is not None and idx < length:
        if transition.type is not None:
            history += str(transition.type.value)
        transition = transition.previous
        idx += 1
    if len(history) == length:
        transDic[label] = history
    transition = transRef
