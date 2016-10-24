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
    domain2ID = {}
    ID2domain = {}
    people2ID = {}
    ID2People = {}
    '''
    Variables of interest:
    <basic> domain
    <for mapping> people2ID, ID2people, ftrID2trainRowID
    <for the ML> trainX, trainX, testX, features
    <for classification> regModel, linearWeights
    '''

    def __init__(self, domainFile, featureFile, trainFile, testFile, outputFile, toReduce=True, model=1):
        self.testFile = testFile
        self.outputFile = outputFile

        self.readDomains(domainFile)
        self.readAllFeatures(featureFile)

        self.getTrainMatrices(trainFile, model=model)

        self.getDimensionalityReductionModel(toReduce=toReduce)
        self.trainRegressionModel(self.trainX_red, self.trainY.toarray())

        # self.predict(accuracyFlag=True, toReduce=toReduce, model=model)

    def readDomains(self, domainFile):
        f = open(domainFile, 'r')
        for idx, line in enumerate(f.readlines()):
            l = line.strip()
            self.domain2ID[l] = idx    # Create an entry for the profession
            self.ID2domain[idx] = l    # Create an entry for the profession
        f.close()

    def readAllFeatures(self, featureFile):
        self.features = None

        with open(featureFile) as data_file:
            data = json.load(data_file, encoding="utf-8")
        for sample in data:
            if sample.keys()[0] not in self.people2ID:
                self.people2ID[sample.keys()[0]] = len(self.people2ID)    # Create an entry for the person
                self.ID2People[len(self.people2ID)-1] = sample.keys()[0]

        self.ftr0 = lil_matrix((len(self.people2ID), len(self.domain2ID)), dtype=float)
        self.ftr1 = lil_matrix((len(self.people2ID), len(self.domain2ID)), dtype=float)

        for sample in data:
            person = sample.keys()[0]
            features = sample[person]

            rowID = self.people2ID[person]
            for country in features:
                if country in self.domain2ID:
                    colID = self.domain2ID[country]
                self.ftr0[rowID,colID] = features[country][0]
                self.ftr1[rowID,colID] = features[country][1]
        self.features = csr_matrix(hstack((self.ftr0, self.ftr1)))
        # self.features = csr_matrix(6*ftr0+ftr1)

    def getTrainMatrices(self, trainFile, model=1):
        trainYdict = {}
        f = codecs.open(trainFile, mode='r', encoding='utf-8')
        for line in f.readlines():
            line = line.strip()
            l = line.split('\t')
            trainYdict[(self.people2ID[l[0]], self.domain2ID[l[1]])] = int(l[2])
        f.close()

        if model == 1:
            # Get mapping from train set to row in matrix
            self.ftrID2trainRowID = {}
            for person in trainYdict:
                if person[0] not in self.ftrID2trainRowID:
                    self.ftrID2trainRowID[person[0]] = len(self.ftrID2trainRowID)

            self.trainX = lil_matrix((len(self.ftrID2trainRowID), self.features.shape[1]), dtype=float)
            self.trainY = lil_matrix((len(self.ftrID2trainRowID), len(self.domain2ID)), dtype=float)
            for idx,loc in enumerate(trainYdict):
                self.trainX[self.ftrID2trainRowID[loc[0]],:] = self.features[loc[0],:]
                self.trainY[self.ftrID2trainRowID[loc[0]],loc[1]] = trainYdict[(loc[0], loc[1])]
        elif model == 2:
            self.trainX = np.zeros((len(trainYdict), 2), dtype=float)
            self.trainY = np.zeros((len(trainYdict), 1), dtype=float)
            for idx,loc in enumerate(trainYdict):
                self.trainX[idx,:] = [self.ftr0[loc[0],loc[1]], self.ftr1[loc[0],loc[1]]]
                self.trainY[idx] = trainYdict[(loc[0], loc[1])]
                try:
                    pass
                    # print self.ID2People[loc[0]], self.ID2domain[loc[1]], self.trainX[idx,:], self.trainY[idx]
                except:
                    print 'EXCEPTION :('
        self.trainX = csr_matrix(self.trainX)
        self.trainY = csr_matrix(self.trainY)

    def getLinRegScore(self, inputTuple, toReduce=True, model=1):
        if not inputTuple[0] in self.people2ID:
            return 0
        else:
            if inputTuple[0] not in self.people2ID:
                return 0
            rowID = self.people2ID[inputTuple[0]]
            colID = self.domain2ID[inputTuple[1]]
            if toReduce:    # Assumed that model=1. toReduce=1 and model=2 not allowed!
                testRow = self.reductionModel.transform(self.features[rowID,:].toarray())
            else:
                if model == 1:
                    testRow = self.features[rowID,:].toarray()
                    return np.dot(testRow, self.linearWeights[:,colID])
                elif model == 2:
                    testRow = [self.ftr0[rowID,colID], self.ftr1[rowID,colID]]
                    return np.dot(testRow, self.linearWeights)

    def predict(self, accuracyFlag=False, toReduce=True, model=1):
        # This is for the train-test split. Given a file with test tuples in it,
        # this function computes the predictions for each of tuples, with optional computation of accuracy
        f1 = codecs.open(self.testFile, 'r', encoding='utf-8')
        f2 = codecs.open(self.outputFile, 'w', encoding='utf-8')

        if accuracyFlag:
            comparison = []
            diff = []

        for idx, line in enumerate(f1.readlines()):
            lineSplit = line.strip().split('\t')
            inputTuple = (lineSplit[0], lineSplit[1])
            score = self.getLinRegScore(inputTuple, toReduce=toReduce, model=model)
            if score < 0:
                score = 0
            elif score > 7:
                score = 7
            if accuracyFlag:
                truth = int(lineSplit[2])
                score = np.ceil(score)

                diff.append(abs(truth-score))
                comparison.append(abs(truth-score)<=2)
                # print inputTuple, score, truth
            f2.write(inputTuple[0]+'\t' + inputTuple[1] +'\t' + str(round(score))+'\n')

        if accuracyFlag:
            accuracy = float(sum(comparison)) / len(comparison)
            print 'Accuracy: ', accuracy, '\nMean absolute error: ', np.mean(diff)

        f1.close()
        f2.close()

    def getDimensionalityReductionModel(self, split=False, toReduce=True):
        trainX = self.trainX.toarray()
        trainY = self.trainY.toarray()

        if split:
            trainX, self.testX, trainY, self.testY = train_test_split(self.trainX, self.trainY, test_size=0.2, random_state=0)

        if toReduce:
            self.reductionModel = PCA(n_components=0.95)
            self.reductionModel.fit(trainX)
            self.trainX_red = self.reductionModel.transform(trainX)
        else:
            self.trainX_red = trainX
        if split:
            self.testX_red = self.reductionModel.transform(self.testX)

    def trainRegressionModel(self, trainX, trainY):
        X_train, X_test, y_train, y_test = train_test_split(trainX, trainY,test_size=0.2,random_state=0)

        # self.regModel = linear_model.LinearRegression()
        # self.regModel = linear_model.Lasso(alpha=0.00001)
        # self.regModel = linear_model.Ridge(alpha=0.01)

        self.regModel.fit(trainX, trainY)
        # self.linearWeights = self.regModel.coef_.transpose()

        self.regModel.fit(X_train, y_train)
        print self.regModel.coef_
        print self.regModel.score(X_test, y_test)

        # print self.linearWeights
        # self.linearWeights = [70,1]

if __name__ == '__main__':
    t0 = time.time()

    # All the files:
    domainFile = '../data/wsdm/nationalities'
    featureFile = '../data/wsdm/nationalities_1020.json'
    trainFile = '../data/wsdm/accuracyTestLinearReg/nationalityTrain.train'
    testFile = '../data/wsdm/accuracyTestLinearReg/nationalityTest.train'
    outputFile = '../data/wsdm/accuracyTestLinearReg/linearRegOutput.txt'

    regObject = NationalitiesLinearRegression(domainFile, featureFile, trainFile, testFile, outputFile,toReduce=False, model=2)

    print 'Time elapsed: ', time.time()-t0