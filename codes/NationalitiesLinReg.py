__author__ = 'Abhishek'
import time
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, hstack
from sklearn.decomposition import PCA
import json
import codecs
from pprint import pprint
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

class NationalitiesLinearRegression():
    domain = {}


    trainPeople = {}
    trainX = {}
    trainY = {}
    trainYdict = {}

    testPeople = {}
    testDict = set()

    def __init__(self, domainFile, trainXFile, trainYFile):
        self.readDomains(domainFile)

        self.readTrainX(trainXFile)
        self.readTrainY(trainYFile)
        self.performLinearRegression()

    def readDomains(self, domainFile):
        f = open(domainFile, 'r')
        for idx, line in enumerate(f.readlines()):
            l = line.strip()
            self.domain[l] = idx    # Create an entry for the profession
        f.close()

    def readTrainY(self, trainFile):
        f = codecs.open(trainFile, mode='r', encoding='utf-8')
        for line in f.readlines():
            line = line.strip()
            l = line.split('\t')

            self.trainYdict[(self.trainPeople[l[0]], self.domain[l[1]])] = int(l[2])
        f.close()

        self.trainY = lil_matrix((len(self.trainPeople), len(self.domain)), dtype=float)
        for idx,loc in enumerate(self.trainYdict):
            self.trainY[loc[0],loc[1]] = self.trainYdict[(loc[0], loc[1])]
        self.trainY = csr_matrix(self.trainY)

    def readTrainX(self, trainXFile):
        with open(trainXFile) as data_file:
            data = json.load(data_file, encoding="utf-8")

        for sample in data:
            if sample.keys()[0] not in self.trainPeople:
                self.trainPeople[sample.keys()[0]] = len(self.trainPeople)    # Create an entry for the person

        trainX0 = lil_matrix((len(self.trainPeople), len(self.domain)), dtype=float)
        trainX1 = lil_matrix((len(self.trainPeople), len(self.domain)), dtype=float)

        for sample in data:
            person = sample.keys()[0]
            features = sample[person]

            rowID = self.trainPeople[person]
            for country in features:
                if country in self.domain:
                    colID = self.domain[country]

                trainX0[rowID,colID] = features[country][0]
                trainX1[rowID,colID] = features[country][1]

        self.trainX = hstack((trainX0, trainX1))
        # self.trainX = trainX1

    def performLinearRegression(self):
        # trainX_Red = self.trainX.toarray()
        # pca = PCA(n_components=0.90)
        # trainX_Red = pca.fit_transform(trainX_Red)

        self.trainX = self.trainX.toarray()
        self.trainY = self.trainY.toarray()

        trainX, self.testX, trainY, self.testY = train_test_split(self.trainX, self.trainY, test_size=0.2, random_state=0)
        # trainX, trainY = self.trainX, self.trainY

        # self.regModel = linear_model.LinearRegression()
        self.regModel = linear_model.Ridge(alpha=0.1)
        self.regModel.fit(trainX, trainY)
        self.linearWeights = self.regModel.coef_.transpose()

    def getScore(self, inputTuple):
        if not inputTuple[0] in self.trainPeople:
            return 0
        else:
            if inputTuple[0] not in self.trainPeople:
                return 0
            rowID = self.trainPeople[inputTuple[0]]
            colID = self.domain[inputTuple[1]]
            return np.dot(self.trainX[rowID,:], self.linearWeights[:,colID])

    def getAccuracy(self, fileName):
        f = codecs.open(fileName, 'r', encoding='utf-8')
        comparison = []
        diff = []
        for idx, line in enumerate(f.readlines()):
            lineSplit = line.strip().split('\t')
            inputTuple = (lineSplit[0], lineSplit[1])
            score = self.getScore(inputTuple)
            if score < 0:
                score = 0
            if score > 7:
                score = 7

            truth = int(lineSplit[2])
            diff.append(abs(truth-score))
            comparison.append(truth-2<=score<=truth+2)
        accuracy = float(sum(comparison)) / len(comparison)
        print accuracy, np.mean(diff)
        print comparison
        f.close()

    def predict(self, inFile, outFile):
        f1 = codecs.open(inFile, 'r', encoding='utf-8')
        f2= codecs.open(outFile, 'w', encoding='utf-8')
        for idx, line in enumerate(f1.readlines()):
            lineSplit = line.strip().split('\t')
            inputTuple = (lineSplit[0], lineSplit[1])
            score = self.getScore(inputTuple)
            if score < 0:
                score = 0
            if score > 7:
                score = 7
            f2.write(inputTuple[0]+'\t' + inputTuple[1] +'\t' + str(round(score))+'\n')
        f2.close()
        f1.close()

if __name__ == '__main__':
    t0 = time.time()
    domainFile = '../data/wsdm/nationalities'
    trainXFile = '../data/wsdm/nationalities.json'
    # trainYFile = '../data/wsdm/nationality.train'
    trainYFile = '../data/wsdm/accuracyTestLinearReg/nationalityTrain.train'

    # domainFile = '../data/wsdm/professions'
    # trainFile = '../data/wsdm/profession.train'

    regObject = NationalitiesLinearRegression(domainFile, trainXFile, trainYFile)
    regObject.predict('../data/wsdm/accuracyTestLinearReg/nationalityTest.train', '../data/wsdm/accuracyTestLinearReg/linearRegOutput.txt')
    # regObject.getAccuracy('../data/wsdm/accuracyTestLinearReg/nationalityTest.train')

    print 'Time elapsed: ', time.time()-t0