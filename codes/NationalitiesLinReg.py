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

        self.readTrainY(trainYFile)
        self.readTrainX(trainXFile)
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
            if l[0] not in self.trainPeople:
                self.trainPeople[l[0]] = len(self.trainPeople)    # Create an entry for the person

            self.trainYdict[(self.trainPeople[l[0]], self.domain[l[1]])] = int(l[2])
        f.close()

        self.trainY = lil_matrix((len(self.trainPeople), len(self.domain)), dtype=float)
        for idx,loc in enumerate(self.trainYdict):
            self.trainY[loc[0],loc[1]] = self.trainYdict[(loc[0], loc[1])]
        self.trainY = csr_matrix(self.trainY)

    def readTrainX(self, trainXFile):
        with open(trainXFile) as data_file:
            data = json.load(data_file, encoding="utf-8")

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
        trainX_Red = self.trainX.toarray()
        pca = PCA(n_components=0.90)
        trainX_Red = pca.fit_transform(trainX_Red)

        trainXtemp, testXtemp, trainYtemp, testYtemp = train_test_split(trainX_Red, self.trainY, test_size=0.2, random_state=0)

        self.regModel = linear_model.LinearRegression()
        self.regModel.fit(self.trainX.toarray(), self.trainY.toarray())

if __name__ == '__main__':
    t0 = time.time()
    domainFile = '../data/wsdm/nationalities'
    trainXFile = '../data/wsdm/nationalities.json'
    trainYFile = '../data/wsdm/nationality.train'

    # domainFile = '../data/wsdm/professions'
    # trainFile = '../data/wsdm/profession.train'

    regObject = NationalitiesLinearRegression(domainFile, trainXFile, trainYFile)

    inputTuple = ('Nikola Tesla', 'United States of America')
    regObject.getScore(inputTuple)

    print 'Time elapsed: ', time.time()-t0