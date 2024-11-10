import os
import nltk
origPath = "C:/Users/stewa/Year 4 Uni/NLP/Coursework/venv"
posPath = "C:/Users/stewa/Year 4 Uni/NLP/Coursework/venv/data/pos"
negPath = "C:/Users/stewa/Year 4 Uni/NLP/Coursework/venv/data/neg"
os.chdir(posPath)

allFileList = []

posFileList = []
negFileList = []

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

def calcTF(term):
    pass

def calcIDF(term):
    pass

posTrain = 0
for file in os.listdir():
    if posTrain < 1600:
        path = posPath + '/' + file
        PosFileRead(path)
        posTrain += 1
    elif posTrain < 1800:
        path = posPath + '/' + file
        EvalFileRead(path)
        posTrain += 1
    else:
        path = posPath + '/' + file
        TestFileRead(path)


'''Collect all the file names ready to read in and add to training lists'''
os.chdir(negPath)
negTrain = 0
for file in os.listdir():
    if negTrain < 1600:
        path = negPath + '/' + file
        NegFileRead(path)
        negTrain += 1
    elif negTrain < 1800:
        path = negPath + '/' + file
        EvalFileRead(path)
        negTrain += 1
    else:
        path = negPath + '/' + file
        TestFileRead(path)



# print(len(posFileList))
# print(len(negFileList))
# print(len(evalList))
# print(len(testList))

'''Tokenise the pieces of text and store the tokens in txt files'''

os.chdir(origPath)
posFile = open('posList.txt', 'w',encoding="utf8")
for review in posFileList:
    posFile.write(review+'\n')
posFile.close()

negFile = open('negList.txt', 'w',encoding="utf8")
for review in negFileList:
    negFile.write(review+'\n')
negFile.close()

evalFile = open("evalList.txt", 'w', encoding="utf8")
for review in evalList:
    evalFile.write(review+'\n')
evalFile.close()

testFile = open("testList.txt", 'w', encoding="utf8")
for review in testList:
    testFile.write(review+'\n')
testFile.close()

'''Test different forms of tokenisation here'''
allTokPos = []
lowerPos = []
for i in posFileList:
    tokens = i.split(' ')
    for token in tokens:
        if '<' not in token and '>' not in token:
            posTokenList.append(token.lower())
            allTokPos.append(token.lower())
    lowerPos.append(posTokenList)
    posTokenList = []

lowerNeg = []
allTokNeg = []
for i in negFileList:
    tokens = i.split(' ')
    for token in tokens:
        if '<' not in token and '>' not in token:
            negTokenList.append(token.lower())
            allTokNeg.append(token.lower())
    lowerNeg.append(negTokenList)
    negTokenList = []


lowerAllTok = []
lowerAllTok.extend(lowerNeg)
lowerAllTok.extend(lowerPos)

'''Test different cutoffs for term frequency'''

fdistPos = nltk.FreqDist(allTokPos)
fdistNeg = nltk.FreqDist(allTokNeg)
# print(type(fdistPos))
# print(fdistPos.most_common())

'''#Extract compositional phrases, possibly by PoS & Constituency parsing or frequently occurring n-grams

 PoS and constituency parsing to extract noun phrases into a list to add to my vocabulary'''

# Extracting all Noun Phrases in the negative reviews
nounPhrases = []
for i in range(len(lowerNeg)):
    tagged = nltk.pos_tag(lowerNeg[i])

    chunker = nltk.RegexpParser("""
                        NP: {<DT>?<JJ>*<NN>} #To extract Noun Phrases
                        P: {<IN>}            #To extract Prepositions
                        V: {<V.*>}           #To extract Verbs
                        PP: {<p> <NP>}       #To extract Prepositional Phrases
                        VP: {<V> <NP|PP>*}   #To extract Verb Phrases
                        """)
    #construct the constituency tree
    tree = chunker.parse(tagged)
    # extract noun phrases
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        myPhrase = ''
        for item in subtree.leaves():
            myPhrase += ' '+item[0]
        nounPhrases.append(myPhrase.strip())

'''INCLUDE IN REPORT (Hard section, boosting features)'''


'''Normalise, TF-IDF and one other method (maybe MRR) ALSO INCLUDE IN REPORT'''



'''Here we run all our experiments, show that our final combination is the best, show table of performance
After this is where we will use our test set'''


#Implement our own NB algorithm INCLUDE CODE