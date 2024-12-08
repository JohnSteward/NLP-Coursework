import os
import re
import nltk
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from nltk.stem.lancaster import LancasterStemmer
import math
from sklearn import metrics
import matplotlib.pyplot as plt
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

# Calculates the IDF of a given word in the training set
def calcIDFWord(term, docList):
    count = 0
    for i in docList:
        if term in i:
            count += 1
    return math.log((len(docList)/(count+1)), 10)

#Function to extract n-grams
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
                # dictionary[st.stem(myPhrase.strip())] = dictionary.get(
                #     st.stem(myPhrase.strip()), 0) + 1
                temp.append(lemmatiser.lemmatize(myPhrase.strip()))
                # temp.append(st.stem(myPhrase.strip()))
        phraseDoc.append(temp)

# Function to perform PoS tagging and constituency parsing
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
            # dictionary[st.stem(myPhrase.strip())] = dictionary.get(st.stem(myPhrase.strip()), 0) + 1
            dictionary[lemmatiser.lemmatize(myPhrase.strip())] = dictionary.get(lemmatiser.lemmatize(myPhrase.strip()), 0) + 1
            # phrases[st.stem(myPhrase.strip())] = phrases.get(st.stem(myPhrase.strip()), 0) + 1
            phrases[lemmatiser.lemmatize(myPhrase.strip())] = phrases.get(lemmatiser.lemmatize(myPhrase.strip()), 0) + 1
            # tempList.append(st.stem(myPhrase.strip()))
            tempList.append(lemmatiser.lemmatize(myPhrase.strip()))
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
            # tempList.append(st.stem(myPhrase.strip()))
            tempList.append(lemmatiser.lemmatize(myPhrase.strip()))
        list.append(tempList)

def StemVocab(dictionary):
    for i in list(dictionary.keys()):
        if i not in stopList and i not in string.punctuation:
            stem = st.stem(i)
            lemma = lemmatiser.lemmatize(i)
            dictionary[lemma] = dictionary.pop(i)

def NaiveBayesTrain(docs, classes):
    # Compile all of the tf-idf values of the training data to use in the Naive Bayes calculation
    posIDF = np.zeros(len(docs[0]))
    negIDF = np.zeros(len(docs[0]))
    for i in range(len(docs)):
        for j in range(len(docs[i])):
            if classes[i]:
                posIDF[j] += docs[i][j]
            else:
                negIDF[j] += docs[i][j]
    return posIDF, negIDF

def NegNaiveBayes(testDoc, negIDF, totalIDF):
    total = 0.5
    for i in range(len(testDoc)):
        if testDoc[i] != 0:
            total *= (testDoc[i]*negIDF[i]) / totalIDF
    return total

def PosNaiveBayes(testDoc, posIDF, totalIDF):
    total = 0.5
    for i in range(len(testDoc)):
        if testDoc[i] != 0:
            total *= (testDoc[i]*posIDF[i]) / totalIDF
    return total

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
os.chdir(origPath)


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

# Tokenising training docs
for i in posFileList:
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
    tokens = word_tokenize(i.lower())
    for token in tokens:
        if token == 'br':
            tokens.remove(token)
        else:
            evalTokList.append(token)
            allTokEval.append(token)
    lowerEval.append(evalTokList)
    evalTokList = []

allTokTest = []
testTokList = []
lowerTest = []

# Tokenising test docs
for i in testList:
    tokens = word_tokenize(i.lower())
    for token in tokens:
        if token == 'br':
            tokens.remove(token)
        else:
            testTokList.append(token)
            allTokTest.append(token)
    lowerTest.append(testTokList)
    testTokList = []


#lowerNeg/pos are split up by document (list of documents that contain a list of words)
lowerAllTok = []
allTok = []
allTok.extend(allTokPos)
allTok.extend(allTokNeg)
lowerAllTok.extend(lowerPos)
lowerAllTok.extend(lowerNeg)
print(allTok)
fDist = nltk.FreqDist(allTok)
# Only use terms that appear more than 75 times and less than 850 times
minFreq = 75
maxFreq = 850
phraseDoc = []
cutoff = {}
allNGrams = {}
n = 3
NGrams(lowerAllTok, allNGrams, n)


# Reducing our vocabulary by removing words that are too frequent or not frequent enough
for i in fDist.most_common():
    if i[1] < maxFreq and i[1] > minFreq:
        cutoff[i[0]] = i[1]
allIDF = {}

# Calculate the IDF for all words in the vocabulary, we do the same for phrases below
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

# extractPhrasesTraining(chunker, lowerAllTok, cutoff, phraseDoc)
# Extracting n-grams
for i in list(allNGrams.keys()):
    if allNGrams[i] > minFreq and allNGrams[i] < maxFreq:
        cutoff[i] = allNGrams[i]
    else:
        del allNGrams[i]
print(allNGrams)


# For PoS tagging
# for i in phrases:
#     if phrases[i] < minFreq or phrases[i] > maxFreq:
#         delItems.append(i)
# for i in delItems:
#     del cutoff[i]
#     del phrases[i]

# Switch to phrases if testing PoS
for i in allNGrams:
    allIDF[i] = calcIDFWord(i, phraseDoc)




print('now stem')
# Stem all words in the dictionary after PoS tagging

StemVocab(cutoff)
StemVocab(allIDF)





trainingIDFVals = []
evalIDFVals = []
testIDFVals = []
tempList = []
# Vectorising the terms in each document, keeping them consistent with each other

# Average length for BM25
avgLen = 0
for i in lowerAllTok:
    avgLen += len(i)
avgLen = avgLen/len(lowerAllTok)
k = 3
b = 0.5
index = 0
for doc in lowerAllTok:
    for i in range(len(doc)):
        if doc[i] not in stopList and doc[i] not in string.punctuation:
            doc[i] = lemmatiser.lemmatize(doc[i])
            # doc[i] = st.stem(doc[i])
    docFreq = []
    for i in cutoff:
        # TF-IDF
        # docFreq.append((doc.count(i)+phraseDoc[index].count(i))*allIDF[i])
        docFreq.append((((doc.count(i)+phraseDoc[index].count(i))*(k+1))/((doc.count(i)+phraseDoc[index].count(i))+k
                                                                          *(1-b + (b*(len(doc))/avgLen))))*allIDF[i])
    trainingIDFVals.append(docFreq)
    index += 1
print('done vectorising training')

# Do the same with the eval set
evalPhrase = []
testPhrase = []
# Extracting all the noun phrases in the test data and comparing to the vocabulary (cutoff)
# extractPhrasesTesting(chunker, lowerEval, evalPhrase)

posIDF, negIDF = NaiveBayesTrain(trainingIDFVals, trainingClasses)
posTotal = 0
negTotal = 0
for i in posIDF:
    posTotal += i
for i in negIDF:
    negTotal += i

index = 0
print('extracted testing')
for doc in lowerEval:
    for i in range(len(doc)):
        if doc[i] not in stopList and doc[i] not in string.punctuation:
            doc[i] = lemmatiser.lemmatize(doc[i])
            # doc[i] = st.stem(doc[i])
        # Remove this part if testing PoS
        if i + n < len(doc):
            phraseInd = 0
            myPhrase = ''
            while phraseInd <= n:
                myPhrase += ' ' + doc[i + phraseInd]
                phraseInd += 1
            tempList.append(lemmatiser.lemmatize(myPhrase.strip()))
            # tempList.append(st.stem(myPhrase.strip()))
        evalPhrase.append(tempList)
        tempList = []

    docFreq = []
    for i in cutoff:
        # TF-IDF
        docFreq.append((doc.count(i)+evalPhrase[index].count(i))*allIDF[i])
        # docFreq.append((((doc.count(i) + evalPhrase[index].count(i)) * (k + 1)) / (
        #             (doc.count(i) + evalPhrase[index].count(i)) + k * (1 - b + (b * (len(doc)) / avgLen)))) * allIDF[i])
    evalIDFVals.append(docFreq)
    index += 1

index = 0
for doc in lowerTest:
    for i in range(len(doc)):
        if doc[i] not in stopList and doc[i] not in string.punctuation:
            doc[i] = lemmatiser.lemmatize(doc[i])
            # doc[i] = st.stem(doc[i])
        # Remove this if testing PoS
        if i + n < len(doc):
            phraseInd = 0
            myPhrase = ''
            while phraseInd <= n:
                myPhrase += ' ' + doc[i + phraseInd]
                phraseInd += 1
            tempList.append(lemmatiser.lemmatize(myPhrase.strip()))
            # tempList.append(st.stem(myPhrase.strip()))
        testPhrase.append(tempList)
        tempList = []
    docFreq = []
    for i in cutoff:
        # TF-IDF
        # docFreq.append((doc.count(i)+testPhrase[index].count(i))*allIDF[i])
        docFreq.append((((doc.count(i) + testPhrase[index].count(i)) * (k + 1)) / (
                    (doc.count(i) + testPhrase[index].count(i)) + k * (1 - b + (b * (len(doc)) / avgLen)))) * allIDF[i])
    testIDFVals.append(docFreq)
    index += 1

# Train our Naive Bayes algorithm
classifier = nb.MultinomialNB()
classifier.fit(trainingIDFVals, trainingClasses)
evalPredicted = classifier.predict(evalIDFVals)
# testPredicted = classifier.predict(testIDFVals)
print(classifier.score(evalIDFVals, evalClasses))
# print(classifier.score(testIDFVals, testClasses))


#Implement our own NB algorithm INCLUDE CODE
# predClasses = []
# for i in testIDFVals:
#     if NegNaiveBayes(i, negIDF, negTotal) >= PosNaiveBayes(i, posIDF, posTotal):
#         predClasses.append(0)
#     else:
#         predClasses.append(1)
#
# predScore = 0
# for i in range(len(predClasses)):
#     if predClasses[i] == testClasses[i]:
#         predScore += 1
# print(predScore/len(predClasses))



#
# confMat = metrics.confusion_matrix(testClasses, predClasses)
# cmDisplay = metrics.ConfusionMatrixDisplay(confusion_matrix=confMat, display_labels=[0,1])
# cmDisplay.plot()
# plt.tight_layout()
# plt.xlabel("True Label")
# plt.ylabel("Predicted Label")
# plt.show()