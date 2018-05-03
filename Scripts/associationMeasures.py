import sys
import xml.etree.ElementTree as ET

from nltk.stem import WordNetLemmatizer

cbtScoresXML = '../mwetoolkit/CBT-scores/candidates-features.xml'
cbtScoresCSV = '../mwetoolkit/CBT-scores/candidates-features.csv'

lppScoresXML = '../mwetoolkit/LPP-scores/candidates-features.xml'
lppScoresCSV = '../mwetoolkit/LPP-scores/candidates-features.csv'

rawCandidates = '../mwetoolkit/CBT-scores/candidates.txt'

# lppLexiconPath = '../Corpora/Lexicons/LPP/lexicon.txt'
lexiconPatternXmlPath = '../mwetoolkit/LPP-patterns.xml'


def createPatternXml(lppLexiconPath, lexiconPatternXmlPath):
    wordnet_lemmatizer = WordNetLemmatizer()
    patternFileTxt = '<?xml version="1.0" encoding="UTF-8"?>'
    patternFileTxt += '<!DOCTYPE dict SYSTEM "dtd/mwetoolkit-patterns.dtd">'
    patternFileTxt += '<patterns>'
    with open(lppLexiconPath, 'r') as lexiconFile:
        for line in lexiconFile:
            line = line[:-1]
            patternFileTxt += '<pat>'
            for w in line.split(' '):
                if w.strip():
                    lemma = wordnet_lemmatizer.lemmatize(w)
                    patternFileTxt += '<w surface="{0}"></w>'.format(lemma)
            patternFileTxt += '</pat>'
    patternFileTxt += '</patterns>'
    with open(lexiconPatternXmlPath, 'w') as patternsFile:
        patternsFile.write(patternFileTxt)


def getScoresFromXML(readingPath, prefix='cbt'):
    mweFreqDic, tokenFreqDic = getDicsFromTarget(
        '../Corpora/Lexicons/targetMWEs.csv', '../Corpora/Lexicons/coca100k.txt')
    tree = ET.parse(readingPath)
    root = tree.getroot()
    result = {}
    for cand in root:
        if cand.tag == 'cand':
            mweLemmaStr, mweStr, posTags, frequency, features = '', '', '', 0, dict()
            for c in cand:
                if c.tag == 'ngram':
                    for w in c:
                        if w.tag == 'w':
                            if 'surface' in w.attrib:
                                mweStr += w.attrib['surface'] + ' '
                            else:
                                mweStr += w.attrib['lemma'] + ' '
                    mweStr = mweStr[:-1]
                if c.tag == 'features':
                    for ch in c:
                        name = ch.attrib['name']
                        value = float(ch.attrib['value'])
                        features[name] = value
            frequency = str(mweFreqDic[mweStr])
            tokenFreqStr = ''
            for token in mweStr.split(' '):
                tokenFreqStr += str(tokenFreqDic[token]) + '-'
            tokenFreqStr = tokenFreqStr[:-1]
            mle = features['mle_' + prefix] if ('mle_' + prefix) in features else''
            dice = features['dice_' + prefix] if ('dice_' + prefix) in features else''
            ll = features['ll_' + prefix] if ('ll_' + prefix in features) else''
            t = features['t_' + prefix] if ('t_' + prefix in features) else''
            pmi = features['pmi_' + prefix] if ('pmi_' + prefix in features) else''
            result['{0} , {1} , {2} , {3} , {4} , {5} , {6}, {7}\n'.
                format(mweStr, frequency, tokenFreqStr, mle, dice, ll, t, pmi)] = True
    with open(readingPath[:-3] + 'csv', 'w') as csvFile:
        csvFile.write(''.join(sorted(result.keys())))


def getCandidatesFromXML(readingPath, writingPath):
    tree = ET.parse(readingPath)
    root = tree.getroot()
    result = ''
    for cand in root:
        if cand.tag == 'cand':
            mweStr, frequency, features = '', 0, dict()
            for c in cand:
                if c.tag == 'ngram':
                    for w in c:
                        if w.tag == 'w':
                            mweStr += w.attrib['lemma'] + ' '
            result += mweStr.strip() + '\n'
    with open(writingPath, 'w') as csvFile:
        csvFile.write(result)


def createCandiateCountXML(path):
    candidId = 1
    mweFreqDic, tokenFreqDic = getDicsFromTarget(
        '../Corpora/Lexicons/targetMWEs.csv', '../Corpora/Lexicons/coca100k.txt')
    candCountTxt = '<?xml version="1.0" encoding="UTF-8"?>'
    candCountTxt += '<!DOCTYPE candidates SYSTEM "dtd/mwetoolkit-candidates.dtd">'
    candCountTxt += '<candidates><meta><corpussize name="cbt" value="450135083"/></meta>'
    nonCand = 0
    for mwe in mweFreqDic:
        if mwe == 'at first ':
            pass
        addCandid, candText = True, ''
        candText += '<cand candid="{0}"><ngram>'.format(candidId)
        candidId += 1
        for token in mwe.split(' '):
            if not token:
                continue
            if token.lower().strip() in tokenFreqDic:
                candText += '<w lemma="{0}" ><freq name="cbt" value="{1}"/></w>'.format(token, tokenFreqDic[token])
            else:
                addCandid = False
        candText += '<freq name="cbt" value="{0}"/></ngram></cand>'.format(mweFreqDic[mwe])
        if addCandid:
            candCountTxt += candText
        else:
            print mwe
            nonCand += 1
    candCountTxt += '</candidates>'
    print nonCand, 'non recognisable expressions'
    with open(path, 'w') as candCountFile:
        candCountFile.write(candCountTxt)


def getDicsFromTarget(lexiconPath, freqPath):
    mweDic, mweTokenDic = dict(), dict()
    with open(lexiconPath, 'r') as f:
        for line in f:
            lineParts = line.strip().lower().split(',')
            mweDic[lineParts[0]] = lineParts[1]
            for token in lineParts[0].split(' '):
                mweTokenDic[token] = 0
    with open(freqPath, 'r') as f:
        for line in f:
            lineParts = line.strip().lower().split('\t')
            if len(lineParts) > 6 and lineParts[1] in mweTokenDic:
                mweTokenDic[lineParts[1]] += int(lineParts[6])
    mweTokenDic['america'] = 149464
    mweTokenDic['pacific'] = 21016
    mweTokenDic['christmas'] = 38673
    print 'MWE DIC LENGTH:', len(mweDic)
    print 'MWE TOKEN DIC LENGTH:', len(mweTokenDic)
    return mweDic, mweTokenDic


reload(sys)
sys.setdefaultencoding('utf8')
if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    # cbtScoresXML = '../mwetoolkit/New-CBT-scores/candidates-features.xml'
    # cbtScoresCSV = '../mwetoolkit/New-CBT-scores/candidates-features.csv'
    createCandiateCountXML('../mwetoolkit/Coca-scores/candidate-counts-new.xml')
    getScoresFromXML('../mwetoolkit/Coca-scores/3/candidates-features.xml')
    # getScoresFromXML(cbtScoresXML, cbtScoresCSV, 'cbt')
    # lppLexiconPath = '/Users/halsaied/PycharmProjects/LePetitPrince/Corpora/LPP/mweLEX.txt'
    # createPatternXml('/Users/halsaied/PycharmProjects/LePetitPrince/Corpora/Lexicons/newLexicon.csv', '/Users/halsaied/PycharmProjects/LePetitPrince/AssociationMeasures/New-patterns.xml')
