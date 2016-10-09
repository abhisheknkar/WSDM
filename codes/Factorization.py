__author__ = 'Abhishek'
import time
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.decomposition import NMF

class FactorizationModel():
    domain = {}
    trainPeople = {}
    trainDict = {}
    testPeople = {}
    testDict = set()

    def __init__(self, domainFile, trainFile, testFile, modelType='professions'):
        self.readDomains(domainFile)

        self.readTrainFile(trainFile)
        self.generateTrainMat()
        self.generateBinaryTrain()

        # self.readTestFile(testFile)
        # self.generateTestMat()

        # self.performNMF()

    def readDomains(self, domainFile):
        f = open(domainFile, 'r')
        for idx, line in enumerate(f.readlines()):
            l = line.strip()
            self.domain[l] = idx    # Create an entry for the profession
        f.close()

    def readTrainFile(self, trainFile):
        f = open(trainFile, 'r')
        for line in f.readlines():
            line = line.strip()
            l = line.split('\t')
            if l[0] not in self.trainPeople:
                self.trainPeople[l[0]] = len(self.trainPeople)    # Create an entry for the person
            self.trainDict[(self.trainPeople[l[0]], self.domain[l[1]])] = int(l[2])
        f.close()

    def readTestFile(self, testFile):
        f = open(testFile, 'r')
        for line in f.readlines():
            line = line.strip()
            l = line.split('\t')
            if l[0] not in self.testPeople:
                self.testPeople[l[0]] = len(self.testPeople)    # Create an entry for the person
            self.testDict.add((self.testPeople[l[0]], self.domain[l[1]]))
        f.close()

    def generateTrainMat(self):
        self.trainMat = lil_matrix((len(self.trainPeople), len(self.domain)), dtype=float)
        for idx,loc in enumerate(self.trainDict):
            self.trainMat[loc[0],loc[1]] = self.trainDict[(loc[0], loc[1])]
        self.trainMat = csr_matrix(self.trainMat)

    def generateTestMat(self):
        self.testMat = lil_matrix((len(self.testPeople), len(self.domain)), dtype=float)
        for idx,loc in enumerate(self.testDict):
            self.testMat[loc[0],loc[1]] = 1
        self.testMat = csr_matrix(self.testMat)

    def generateBinaryTrain(self):
        self.binaryTrainMat = lil_matrix((len(self.trainPeople), len(self.domain)))
        for idx,loc in enumerate(self.trainDict):
            self.binaryTrainMat[loc[0],loc[1]] = 1
        self.trainMat = csr_matrix(self.trainMat)

    def performNMF(self):
        model = NMF()
        model.fit(self.testMat)
        print model.reconstruction_err_

if __name__ == '__main__':
    t0 = time.time()
    domainFile = '../data/wsdm/professions'
    trainFile = '../data/wsdm/profession.train'
    testFile = '../data/wsdm/profession.kb'
    FactorizationModel(domainFile, trainFile, testFile)

    print 'Time elapsed: ', time.time()-t0

    # self.trainMat
    # self.testMat
    # self.binaryMat