__author__ = 'Abhishek'
import time
import nltk
from nltk.corpus import stopwords
import wikipedia, string
import os
from nltk.stem.wordnet import WordNetLemmatizer
import operator
import cPickle as pickle

class WikiTransactions():
    def __init__(self, transactionFolder='../data/wsdm/similarWordsProfession/professionTransactions/', professionsFile='../data/wsdm/professions', offlinePath='../data/wsdm/offlineProfessions.pickle', append=True, saveOffline=False):
        # Initializing stuff
        self.lmtzr = WordNetLemmatizer()
        self.cachedStopWords = stopwords.words("english")
        self.cachedPunctuations = set(string.punctuation)
        self.cachedPunctuations.remove('-')
        self.transactionFolder = transactionFolder
        self.professions = self.getProfessions(professionsFile)
        self.failedProfessions = []
        self.offlinePages = {}
        self.offlinePath = offlinePath
        self.getEquivalentProfessions()

        # Getting transactions
        if saveOffline == True:
            self.saveOfflinePages()

        self.getAllTransactions(offlineFlag=True)

        # Handling failed professions
        print 'Failed professions: ', self.failedProfessions
        f = open('../data/wsdm/similarWordsProfession/failedProfessions.txt','w')
        for profession in self.failedProfessions:
            f.write(profession+'\n')

        # self.getNLTKsimilar()

    def getProfessions(self, file):
        professsions = []
        f = open(file, 'r')
        for line in f.readlines():
            professsions.append(line.strip())
        return professsions

    def getWikiPage(self, query, offlineFlag):
        print 'Processing wikipedia page for', query
        try:
            if offlineFlag:
                wikiPage = self.offlinePages[query]
            else:
                wikiPage = wikipedia.page(query)
            return wikiPage
        except:
            self.failedProfessions.append(query)
            return None

    def removeStopWords(self, content):
        removed = ' '.join([word for word in content.split() if word not in self.cachedStopWords])
        return removed

    def removePunctuation(self, text):
        removed = ''.join(ch for ch in text if ch not in self.cachedPunctuations)
        return removed

    def getAllTransactions(self, offlineFlag=True):
        if offlineFlag:
            with open(self.offlinePath, 'rb') as handle:
                self.offlinePages = pickle.load(handle)

        if not os.path.isdir(self.transactionFolder):
            os.mkdir(self.transactionFolder)
        for profession in self.professions:
            professionTransactionPath = self.transactionFolder+profession+'.txt'
            if not os.path.exists(professionTransactionPath):
                f = open(professionTransactionPath, 'w')
                page = self.getWikiPage(profession, offlineFlag=offlineFlag)
                if page != None:
                    transactions = self.getTransactionsMovingWindow(page.content)
                    for transaction in transactions:
                        f.write(transaction+'\n')
                f.close()

    def getTransactionsMovingWindow(self, content, window=10, overlap=5):
        # Gets a moving window of transactions
        # print ' the ' in content
        # content = self.removePunctuation(self.removeStopWords(content)).lower().encode('utf-8')
        content = content.lower()
        content = self.removePunctuation(content)
        content = self.removeStopWords(content).encode('utf-8')

        # print ' the ' in content
        wordList = content.split()
        contentLemmatized = []
        for word in content.split():
            try:
                contentLemmatized.append(self.lmtzr.lemmatize(word))
            # if word != contentLemmatized[-1]:
            #     print word, contentLemmatized[-1]
            except:
                pass
        transactions = []
        start = 0
        while start+window < len(contentLemmatized)-1:
            transactions.append(' '.join(contentLemmatized[start+overlap:start+overlap+window]))
            start += overlap
        return transactions

    def saveOfflinePages(self):
        with open(self.offlinePath, 'rb') as handle:
            self.offlinePages = pickle.load(handle)

        for profession in self.professions:
            if profession not in self.offlinePages:
                print 'Saving wikipedia page for', profession
                try:
                    if profession in self.equivalences:
                        self.offlinePages[profession] = wikipedia.page(self.equivalences[profession])
                    else:
                        self.offlinePages[profession] = wikipedia.page(profession)
                except:
                    self.failedProfessions.append(profession)
        with open(self.offlinePath, 'wb') as handle:
            pickle.dump(self.offlinePages, handle)

    def getNLTKsimilar(self, offlineFlag=True):
        if offlineFlag:
            with open(self.offlinePath, 'rb') as handle:
                self.offlinePages = pickle.load(handle)

        for profession in self.offlinePages:
            content = self.offlinePages[profession].content
            wordList = content.split()
            text = nltk.Text(word.lower() for word in wordList)
            print profession
            print text.similar(profession.lower())

    def getEquivalentProfessions(self, failedFile='../data/wsdm/similarWordsProfession/failedProfessionsMod.txt'):
        f = open(failedFile,'r')
        self.equivalences = {}
        for line in f.readlines():
            lsplit = line.strip().split('\t')
            self.equivalences[lsplit[0]] = lsplit[1]

class Apriori():
    def __init__(self, databaseFile, minsup=20, relativeMinSup=0.01, setRelative=False):
        f = open(databaseFile, 'r')
        self.database = []
        for line in f.readlines():
            items = line.split()
            self.database.append(sorted(items))
        self.reverse_index = self.make_reverse_index()
        if setRelative:
            self.minsup = relativeMinSup*len(self.database)
        else:
            self.minsup = minsup
        self.L = []
        self.Lcounts = {}

    def find_frequent1_itemsets(self):
        itemCounts = {}
        # for transaction in self.database:
        #     for item in transaction:
        #         if item not in itemCounts:
        #             itemCounts[item] = 1
        #         else:
        #             itemCounts[item] += 1

        for item in self.reverse_index:
            if len(self.reverse_index[item])>=self.minsup:
                itemCounts[item] = len(self.reverse_index[item])

        frequentOnes = []
        for item in itemCounts:
            frequentOnes.append([item])

        self.Lcounts[1] = itemCounts
        return frequentOnes

    def apriori_gen(self, Lprev):
        C = []  # New candidates
        for idx1,item1 in enumerate(Lprev):
            for idx2,item2 in enumerate(Lprev):
                if idx1 < idx2:
                    if item1[:-1] != item2[:-1]:
                        continue
                    else:
                        toAppend = sorted([item1[-1],item2[-1]])
                        newCandidate = item1[:-1]+toAppend
                        if not self.has_infrequent_subset(newCandidate):
                            C.append(newCandidate)
        return C

    def has_infrequent_subset(self, candidateK):
        for idx, element in enumerate(candidateK):
            subset = candidateK[0:idx]+candidateK[idx+1:]

    def make_reverse_index(self):
        reverse_index = {}
        for idx,tran in enumerate(self.database):
            for item in tran:
                if item not in reverse_index:
                    reverse_index[item] = set()
                reverse_index[item].add(idx)
        return reverse_index

    def execute(self):
        self.L.append(self.find_frequent1_itemsets()) # L1
        while(1):
            C = self.apriori_gen(self.L[-1])
            if len(self.L[-1])==0:
                return self.L
            k = len(self.L[-1][0])+1
            self.Lcounts[k] = {}
            for idx, candidateSet in enumerate(C):
                occurence_sets = []
                for item in candidateSet:
                    occurence_sets.append(set(self.reverse_index[item]))
                common_occurence = set.intersection(*occurence_sets) # List expansion!
                self.Lcounts[k][tuple(candidateSet)] = len(common_occurence)
                # if len(self.L) == 2 and len(common_occurence) >= self.minsup:
                #     print candidateSet, common_occurence

            # for idx,transaction in enumerate(self.database):
                # for candidate in C:
                    # if set(candidate)<=set(transaction):
                    #     if tuple(candidate) not in self.Lcounts[k]:
                    #         self.Lcounts[k][tuple(candidate)] = 0
                    #     self.Lcounts[k][tuple(candidate)] += 1

            Lnew = []
            toPop = []
            for subset in self.Lcounts[k]:
                if self.Lcounts[k][subset] >= self.minsup:
                    Lnew.append(list(subset))
                else:
                    toPop.append(subset)
            for element in toPop:
                self.Lcounts[k].pop(element)
            if len(Lnew) == 0:
                break
            else:
                self.L.append(Lnew)

        return self.L

    def getConfidence(self, subset1, subset2):
        subset3 = subset1+subset2
        subset1, subset2, subset3 = sorted(subset1), sorted(subset2), sorted(subset3)

        if tuple(subset3) in self.Lcounts[len(subset3)]:
            if tuple(subset1) in self.Lcounts[len(subset1)]:
                print self.Lcounts[len(subset3)][tuple(subset3)], '/', self.Lcounts[len(subset1)][tuple(subset1)], '='
                return float(self.Lcounts[len(subset3)][tuple(subset3)]) / self.Lcounts[len(subset1)][tuple(subset1)]
        else:
            return 0

    def getClosedPatterns(self):
        self.closedPatterns = {}

        if len(self.Lcounts[len(self.Lcounts)]) == 0:
            maxLevel = len(self.Lcounts) - 1
        else:
            maxLevel = len(self.Lcounts)
        for idx in range(maxLevel):
            if idx == maxLevel-1:
                for itemset in self.Lcounts[maxLevel]:
                    self.closedPatterns[self.itemSet2Tuple(itemset)] = self.Lcounts[maxLevel][itemset]
            else:
                # Check if each subset of the current level is a member of any subset of the next level
                for itemset in self.Lcounts[idx+1]:
                    closedFlag = True
                    if type(itemset) == str:
                        setItemset = set()
                        setItemset.add(itemset)
                    else:
                        setItemset = set(itemset)
                    for bigItemset in self.Lcounts[idx+2]:
                        if setItemset <= set(bigItemset):
                            if self.Lcounts[idx+1][itemset] == self.Lcounts[idx+2][bigItemset]:
                                closedFlag = False
                                break
                    if closedFlag == True:
                        self.closedPatterns[self.itemSet2Tuple(itemset)] = self.Lcounts[idx+1][itemset]
        return self.closedPatterns

    def getMaxPatterns(self):
        self.maxPatterns = {}

        if len(self.Lcounts[len(self.Lcounts)]) == 0:
            maxLevel = len(self.Lcounts) - 1
        else:
            maxLevel = len(self.Lcounts)
        for idx in range(maxLevel):
            if idx == maxLevel-1:
                for itemset in self.Lcounts[maxLevel]:
                    self.maxPatterns[self.itemSet2Tuple(itemset)] = self.Lcounts[maxLevel][itemset]
            else:
                # Check if each subset of the current level is a member of any subset of the next level
                for itemset in self.Lcounts[idx+1]:
                    maxFlag = True
                    if type(itemset) == str:
                        setItemset = set()
                        setItemset.add(itemset)
                    else:
                        setItemset = set(itemset)
                    for bigItemset in self.Lcounts[idx+2]:
                        if setItemset <= set(bigItemset):
                            maxFlag = False
                            break
                    if maxFlag == True:
                        self.maxPatterns[self.itemSet2Tuple(itemset)] = self.Lcounts[idx+1][itemset]
        return self.maxPatterns

    def itemSet2Tuple(self, itemSet):
        if type(itemSet) == str:
            return itemSet
        else:
            return tuple(itemSet)

class FrequentPatterns():
    def __init__(self, inFolder='../data/wsdm/similarWordsProfession/professionTransactions/', outFolderFrequent='../data/wsdm/similarWordsProfession/frequent/', outFolderMax='../data/wsdm/similarWordsProfession/max/'):
        self.getFrequentPatterns(inFolder=inFolder, outFolder=outFolderFrequent)
        self.getMaxPatterns(inFolder=inFolder, outFolder=outFolderMax)
        self.getTopKSimilar()

    def getFrequentPatterns(self, inFolder, outFolder, relativeMinsup=0.01):
        print 'Getting Frequent Patterns'
        files = os.listdir(inFolder)
        if not os.path.isdir(outFolder):
            os.mkdir(outFolder)
        for file in files:
            print 'Getting frequent patterns for: ', file
            f1 = open(outFolder +file+'.patterns','w')
            apriori = Apriori(inFolder+file, relativeMinSup=relativeMinsup)
            apriori.execute()
            # Sort the frequent patterns
            if len(apriori.Lcounts)==0:
                continue
            allPatterns = apriori.Lcounts[1].copy()
            for count in range(len(apriori.Lcounts)-1):
                allPatterns.update(apriori.Lcounts[count+2])
            sortedPatterns = sorted(allPatterns.items(), key=operator.itemgetter(1),reverse=True)
            for pattern in sortedPatterns:
                if type(pattern[0]) is str:
                    patternString = pattern[0]
                else:
                    patternString = ' '.join(pattern[0])
                f1.write(str(pattern[1])+ ' '+ patternString+'\n')
            f1.close()

    def getMaxPatterns(self, inFolder, outFolder, relativeMinsup=0.01):
        print 'Getting Max Patterns'
        files = os.listdir(inFolder)
        if not os.path.isdir(outFolder):
            os.mkdir(outFolder)
        for file in files:
            print 'Getting max patterns for: ', file
            f1 = open(outFolder +file+'.max','w')
            apriori = Apriori(inFolder+file, relativeMinSup=relativeMinsup)
            apriori.execute()

            maxPatterns = apriori.getMaxPatterns()
            # Sort the frequent patterns
            sortedPatterns = sorted(maxPatterns.items(), key=operator.itemgetter(1),reverse=True)
            for pattern in sortedPatterns:
                if type(pattern[0]) is str:
                    patternString = pattern[0]
                else:
                    patternString = ' '.join(pattern[0])
                f1.write(str(pattern[1])+ ' '+ patternString+'\n')
            f1.close()

    def getTopKSimilar(self, k=10, professionsFile='../data/wsdm/professions', folder='../data/wsdm/similarWordsProfession/max/', outFile='../data/wsdm/similarWordsProfession/topKSimilar.txt'):
        print 'Getting Similar Words'
        similarWords = {}
        files = os.listdir(folder)
        f1 = open(outFile,'w')
        for file in files:
            profession = file[0:file.index('.')]
            print 'Getting similar patterns for: ', file
            f2 = open(folder+file, 'r')
            left = k
            for line in f2.readlines():
                if left > 0:
                    word = line.strip().split()[1]
                    # print nltk.tag.pos_tag([word])
                    tag = nltk.tag.pos_tag([word])
                    if tag[0][1] == 'NN':
                        if profession not in similarWords:
                            similarWords[profession] = []
                        similarWords[profession].append(' '.join(line.strip().split()[1:]))
                        left -= 1
                else:
                    break
            f2.close()
        sortedSimilar = sorted(similarWords.items(), key=operator.itemgetter(0))
        for element in sortedSimilar:
            f1.write(element[0]+'\t')
            f1.write('\t'.join(element[1])+'\n')
        f1.close()

    def getProfessions(self, file):
        professsions = []
        f = open(file, 'r')
        for line in f.readlines():
            professsions.append(line.strip())
        return professsions

if __name__ == '__main__':
    t0 = time.time()

    # tx = WikiTransactions(professionsFile='../data/wsdm/professions', saveOffline=False)
    fp = FrequentPatterns()

    print 'Time elapsed:', time.time()-t0