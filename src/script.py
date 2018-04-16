import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


def clearSent(sent):
    res = ''
    tokens = sent.split(' ')
    for t in tokens:
        if not t or any(ch.isdigit() for ch in t):
            pass
        else:
            res += t + ' '
    return res


def rawToConllu(sents):
    conlluTxt = ''
    wordnet_lemmatizer = WordNetLemmatizer()
    for line in sents:
        tokens = tokenize(line[:-1])
        posTags = nltk.pos_tag(tokens)
        lemmaList = []
        for tag in posTags:
            if getWordnetPos(tag[1]):
                lemmaList.append(wordnet_lemmatizer.lemmatize(tag[0], pos=getWordnetPos(tag[1])))
            else:
                lemmaList.append(wordnet_lemmatizer.lemmatize(tag[0]))
        for j in range(len(tokens)):
            conlluTxt += '{0}\t{1}\t{2}\t_\t{3}\t'.format(j + 1, tokens[j], lemmaList[j].lower(),
                                                          posTags[j][1]) + '_\t' * 4 + '_\n'
        conlluTxt += '\n'
    return conlluTxt


def rawToMWEFile(sents):
    conlluTxt = ''
    for line in sents:
        tokens = tokenize(line[:-1])
        for j in range(len(tokens)):
            conlluTxt += '{0}\t{1}\t_\t_\n'.format(j + 1, tokens[j])
        conlluTxt += '\n'
    return conlluTxt


def getWordnetPos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ

    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def tokenize(str):
    tokens = word_tokenize(str)
    return tokens
    # tokens = str.split(' ')
    # for token in tokens:
    #     if not token.strip():
    #         tokens.remove(token)
    #
    # return tokens


def AIWCsvToTextBrut(path):
    idx, res = 0, ''
    with open(path, 'r+') as f:
        for line in f:
            if idx == 0:
                idx += 1
                continue
            lineCells = line.split(',')
            if lineCells[1]:
                res += lineCells[1] + ' '
            else:
                res += '\n'
    print res


def AIWToCoNLL(path, newPath):
    sents = []
    with open(path, 'r+') as f:
        for line in f:
            sents.append(line)
    coNLL = rawToConllu(sents)
    with open(newPath, 'w') as f:
        f.write(coNLL)


if __name__ == '__main__':
    # AIWCsvToTextBrut('/Users/halsaied/Downloads/LPP-Simple/Corpora/AIW/DownRabbit.csv')
    AIWToCoNLL('/Users/halsaied/Downloads/LPP-Simple/Corpora/AIW/AIW.brut.gold',
               '/Users/halsaied/Downloads/LPP-Simple/Corpora/AIW/AIW.conll')

    # sent = 'you 0 1 must 0 1 see 0 1 to -0.80526 3 it 0 1 that 0 2 you 0 2 regularly 0 1 pull 5.1762 2 out 0 1 the 0 2 baobabs 0 1 as 0 2 soon 0 1 as 0 2 they 0 1 can 0 1 be 0 1 told 0 2 apart 6.7577 1 from 0 1 the 0 2 Rose 0 4 bushes 0 1 to 0 2 which 0 2 they 0 1 look 0 1 very 0 1 similar 0 2 when 0 2 they 0 1 re 0 1 very 0 22 young'
    # sent2 = '0 so 0 1 so 0 i 0 2 I 0 think 0 1 thought 0 a 0 1 a 4.3121 lot -0.3657 2 lot 0 about 0 1 about 0 the 0 1 the 0 adventure 0 2 adventures 0 of 0 1 of 0 the 0 1 the 0 jungle 0 7 jungle 0 and 0 1 and 0 in 0 1 in 0.30383 turn 3.485 3 turn 0 i 0 2 I 0 manage 0 3 managed 0 with 0 1 with 0 a 0 1 a 0 coloured 0 1 coloured 0 pencil 0 2 pencil 0 to 0 1 to 0 make 0 1 make 0 my 0 1 my 0 first 0 1 first 0 drawing 0 10 drawing'
    # sent3 = '0 2 you 0 1 must 0 1 see 0 1 to -0.80526 3 it 0 1 that 0 2 you 0 2 regularly 0 1 pull 5.1762 2 out 0 1 the 0 2 baobabs 0 1 as 0 2 soon 0 1 as 0 2 they 0 1 can 0 1 be 0 1 told 0 2 apart 6.7577 1 from 0 1 the 0 2 Rose 0 4 bushes 0 1 to 0 2 which 0 2 they 0 1 look 0 1 very 0 1 similar 0 2 when 0 2 they 0 1 re 0 1 very 0 22 young'
    # sent1 = clearSent(sent)
    # sent3 = clearSent(sent3)
    # sent2 = clearSent(sent2)
    # sent2Tmp = ''
    # tokens2 = sent2.split(' ')
    # idx = 0
    # for t in tokens2:
    #     if idx % 2 == 0:
    #         sent2Tmp += t + ' '
    #     idx += 1
    # sents = [sent1, sent2Tmp, sent3]
    # print rawToConllu(sents)
    # print rawToMWEFile(sents)
