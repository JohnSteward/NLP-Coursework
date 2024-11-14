import os
import re
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from nltk.stem.lancaster import LancasterStemmer
import math
import sklearn.naive_bayes as nb
origPath = "C:/Users/stewa/Year 4 Uni/NLP/Coursework/NLP-Coursework/venv"
posPath = "C:/Users/stewa/Year 4 Uni/NLP/Coursework/NLP-Coursework/venv/data/pos"
negPath = "C:/Users/stewa/Year 4 Uni/NLP/Coursework/NLP-Coursework/venv/data/neg"

# origPath = "C:/Users/John Steward/Work/NLP/NLP-Coursework/venv"
# posPath = "C:/Users/John Steward/Work/NLP/NLP-Coursework/venv/data/pos"
# negPath = "C:/Users/John Steward/Work/NLP/NLP-Coursework/venv/data/neg"

os.chdir(posPath)

allFileList = []

posFileList = []
negFileList = []
trainingClasses = []

evalList = []
evalClasses = []
testList = []
testClasses = []

posTokenList = []
negTokenList = []

#Reads all the Positive training reviews and adds them to a list ready for tokenisation
def PosFileRead(path):
    with open(path, 'r',encoding="utf8") as f:
        fileContent = f.read()
        posFileList.append(fileContent)
        f.close()
#Reads all negative training reviews and adds them to a list
def NegFileRead(path):
    with open(path, 'r',encoding="utf8") as f:
        fileContent = f.read()
        negFileList.append(fileContent)
        f.close()

#Reads 400 more files after getting training data to use for evaluation
def EvalFileRead(path):
    with open(path, 'r',encoding="utf8") as f:
        fileContent = f.read()
        evalList.append(fileContent)
        f.close()

#Reads the last 400 files to use for final testing
def TestFileRead(path):
    with open(path, 'r',encoding="utf8") as f:
        fileContent = f.read()
        testList.append(fileContent)
        f.close()

def calcIDFWord(term, docList):
    # Need to include phrases in this
    count = 0
    for i in docList:
        if term in i:
            count += 1
    return math.log((len(docList)/(count+1)), 10)

def NGrams(docList, dictionary, n):
    for doc in docList:
        temp = []
        for i in range(len(docList)):
            if i + n < len(doc):
                index = 0
                myPhrase = ''
                while index < n:
                    myPhrase += ' ' + doc[i+index]
                    index += 1
                dictionary[lemmatiser.lemmatize(myPhrase.strip())] = dictionary.get(lemmatiser.lemmatize(myPhrase.strip()), 0) + 1
                temp.append(lemmatiser.lemmatize(myPhrase.strip()))
        phraseDoc.append(temp)


def extractPhrasesTraining(chunker, docList, dictionary, list):
    for doc in docList:
        tempList = []
        tagged = nltk.pos_tag(doc)
        # construct the constituency tree
        tree = chunker.parse(tagged)
        # extract noun phrases
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            myPhrase = ''
            for item in subtree.leaves():
                myPhrase += ' ' + item[0]
            dictionary[st.stem(myPhrase.strip())] = dictionary.get(st.stem(myPhrase.strip()), 0) + 1
            phrases[st.stem(myPhrase.strip())] = phrases.get(st.stem(myPhrase.strip()), 0) + 1
            tempList.append(st.stem(myPhrase.strip()))
        list.append(tempList)

def extractPhrasesTesting(chunker, docList, list):
    for doc in docList:
        tempList = []
        tagged = nltk.pos_tag(doc)
        # construct the constituency tree
        tree = chunker.parse(tagged)
        # extract noun phrases
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            myPhrase = ''
            for item in subtree.leaves():
                myPhrase += ' ' + item[0]
            tempList.append(st.stem(myPhrase.strip()))
        list.append(tempList)

def StemVocab(dictionary):
    for i in list(dictionary.keys()):
        if i not in stopList and i not in string.punctuation:
            stem = st.stem(i)
            lemma = lemmatiser.lemmatize(i)
            dictionary[lemma] = dictionary.pop(i)

def NaiveBayes():
    pass

posTrain = 0
for file in os.listdir():
    if posTrain < 1600:
        path = posPath + '/' + file
        PosFileRead(path)
        posTrain += 1
        trainingClasses.append(1)
    elif posTrain < 1800:
        path = posPath + '/' + file
        EvalFileRead(path)
        evalClasses.append(1)
        posTrain += 1
    else:
        path = posPath + '/' + file
        testClasses.append(1)
        TestFileRead(path)


'''Collect all the file names ready to read in and add to training lists'''
os.chdir(negPath)
negTrain = 0
for file in os.listdir():
    if negTrain < 1600:
        path = negPath + '/' + file
        NegFileRead(path)
        trainingClasses.append(0)
        negTrain += 1
    elif negTrain < 1800:
        path = negPath + '/' + file
        EvalFileRead(path)
        evalClasses.append(0)
        negTrain += 1
    else:
        path = negPath + '/' + file
        testClasses.append(0)
        TestFileRead(path)





'''Tokenise the pieces of text and store the tokens in txt files'''
# os.chdir(origPath)
# posFile = open('posList.txt', 'w',encoding="utf8")
# for review in posFileList:
#     posFile.write(review+'\n')
# posFile.close()
#
# negFile = open('negList.txt', 'w',encoding="utf8")
# for review in negFileList:
#     negFile.write(review+'\n')
# negFile.close()
#
# evalFile = open("evalList.txt", 'w', encoding="utf8")
# for review in evalList:
#     evalFile.write(review+'\n')
# evalFile.close()
#
# testFile = open("testList.txt", 'w', encoding="utf8")
# for review in testList:
#     testFile.write(review+'\n')
# testFile.close()
'''Test different forms of tokenisation here'''
allTokPos = []
lowerPos = []
stopList = set(stopwords.words('english'))
lemmatiser = WordNetLemmatizer()
st = LancasterStemmer()
for i in posFileList:
    # tokens = [st.stem(word) for word in word_tokenize(i.lower()) if not word in stopList and
    #           not word in string.punctuation]
    tokens = word_tokenize(i.lower())
    for token in tokens:
        if token == 'br':
            tokens.remove(token)
        else:
            posTokenList.append(token)
            allTokPos.append(token)
    lowerPos.append(posTokenList)
    posTokenList = []

lowerNeg = []
allTokNeg = []
for i in negFileList:
    # tokens = [st.stem(word) for word in word_tokenize(i.lower()) if not word in stopList and
    #           not word in string.punctuation]
    tokens = word_tokenize(i.lower())
    for token in tokens:
        if token == 'br':
            tokens.remove(token)
        else:
            negTokenList.append(token)
            allTokNeg.append(token)
    lowerNeg.append(negTokenList)
    negTokenList = []

# getting tokens for the evaluation set
allTokEval = []
evalTokList = []
lowerEval = []
for i in evalList:
    # tokens = [st.stem(word) for word in word_tokenize(i.lower()) if not word in stopList and
    #           not word in string.punctuation]
    tokens = word_tokenize(i.lower())
    for token in tokens:
        if token == 'br':
            tokens.remove(token)
        else:
            evalTokList.append(token)
            allTokEval.append(token)
    lowerEval.append(evalTokList)
    evalTokList = []


#lowerNeg/pos are split up by document (list of documents that contain a list of words)
lowerAllTok = []
allTok = []
allTok.extend(allTokPos)
allTok.extend(allTokNeg)
lowerAllTok.extend(lowerPos)
lowerAllTok.extend(lowerNeg)
print(allTok)
'''Test different cutoffs for term frequency'''
fDist = nltk.FreqDist(allTok)
minFreq = 75
maxFreq = 850
# Only use terms that appear more than 50 times and less than 1000
phraseDoc = []
cutoff = {}
allNGrams = {}
n = 3
NGrams(lowerAllTok, allNGrams, n)


for i in fDist.most_common():
    if i[1] < maxFreq and i[1] > minFreq:
        cutoff[i[0]] = i[1]
# Calculating IDF of singular words, will do so with phrases later
allIDF = {}

for i in cutoff:
    allIDF[i] = calcIDFWord(i, lowerAllTok)


# Extracting all Noun Phrases in reviews
chunker = nltk.RegexpParser("""
                        NP: {<DT>?<JJ>*<NN>} #To extract Noun Phrases
                        P: {<IN>}            #To extract Prepositions
                        V: {<V.*>}           #To extract Verbs
                        PP: {<p> <NP>}       #To extract Prepositional Phrases
                        VP: {<V> <NP|PP>*}   #To extract Verb Phrases
                        """)



delItems = []

phrases = {}

print('now extract phrases')
#extractPhrasesTraining(chunker, lowerAllTok, cutoff, phraseDoc)
'''Extract compositional phrases, possibly by PoS & Constituency parsing or frequently occurring n-grams

 PoS and constituency parsing to extract noun phrases into a list to add to my vocabulary'''
for i in list(allNGrams.keys()):
    if allNGrams[i] > minFreq and allNGrams[i] < maxFreq:
        cutoff[i] = allNGrams[i]
    else:
        del allNGrams[i]
print(allNGrams)
# for i in phrases:
#     if phrases[i] < minFreq or phrases[i] > maxFreq:
#         delItems.append(i)
# for i in delItems:
#     del cutoff[i]
#     del phrases[i]


for i in allNGrams:
    allIDF[i] = calcIDFWord(i, phraseDoc)




print('now stem')
# Stem all words in the dictionary after PoS tagging

StemVocab(cutoff)
StemVocab(allIDF)

'''INCLUDE IN REPORT (Hard section, boosting features)'''

'''Normalise, TF-IDF and one other method (maybe BM-25) ALSO INCLUDE IN REPORT'''


trainingIDFVals = []
evalIDFVals = []
tempList = []
# Vectorising the terms in each document, keeping them consistent with each other


index = 0
for doc in lowerAllTok:
    for i in range(len(doc)):
        if doc[i] not in stopList and doc[i] not in string.punctuation:
            doc[i] = lemmatiser.lemmatize(doc[i])
        # if i + n < len(doc):
        #     phraseInd = 0
        #     myPhrase = ''
        #     while phraseInd <= n:
        #         myPhrase += ' ' + doc[i + phraseInd]
        #         phraseInd += 1
        #     tempList.append(myPhrase.strip())
        # phraseDoc.append(tempList)
        # tempList = []
    docFreq = []
    for i in cutoff:
        docFreq.append((doc.count(i)+phraseDoc[index].count(i))*allIDF[i])
    trainingIDFVals.append(docFreq)
    index += 1
print('done vectorising training')

# Do the same with the eval set
evalPhrase = []
# Extracting all the noun phrases in the test data and comparing to the vocabulary (cutoff)
#extractPhrasesTesting(chunker, lowerEval, evalPhrase)

index = 0
print('extracted testing')
for doc in lowerEval:
    for i in range(len(doc)):
        if doc[i] not in stopList and doc[i] not in string.punctuation:
            doc[i] = lemmatiser.lemmatize(doc[i])
        if i + n < len(doc):
            phraseInd = 0
            myPhrase = ''
            while phraseInd <= n:
                myPhrase += ' ' + doc[i + phraseInd]
                phraseInd += 1
            tempList.append(lemmatiser.lemmatize(myPhrase.strip()))
        evalPhrase.append(tempList)
        tempList = []
    docFreq = []
    for i in cutoff:
        docFreq.append((doc.count(i)+evalPhrase[index].count(i))*allIDF[i])
    evalIDFVals.append(docFreq)
    index += 1

'''Here we run all our experiments, show that our final combination is the best, show table of performance
After this is where we will use our test set'''
print('done eval vectorising')
classifier = nb.MultinomialNB()
# For each doc, need to have the same word vector, but obv with different values (can use tf-idf or just the word count)
classifier.fit(trainingIDFVals, trainingClasses)
print('done fitting')
print(classifier.score(evalIDFVals, evalClasses))
#Implement our own NB algorithm INCLUDE CODE