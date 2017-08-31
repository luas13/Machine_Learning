# This piece of software is bound by The MIT License (MIT)
# Copyright (c) 2017 Abhishek Kumar
# Code written by : Abhishek Kumar
# Email ID : akuma151@asu.edu

from scipy.spatial import distance
import operator
from scipy import spatial
import numpy as np
from random import randint


def cosine_distance(matrix1, matrix2):
    return distance.cdist(matrix1, matrix2, 'cosine')


class kNN(object):
    def __init__(self, mat):
        self.evaldata = mat['evaldata']
        self.testdata = mat['testdata']
        self.testlabels = mat['testlabels']
        self.traindata = mat['traindata']
        self.trainlabels = mat['trainlabels']
        train_dis_table = cosine_distance(self.traindata, self.traindata)
        self.train_dis_index_table = np.argsort(train_dis_table, axis=1)
        test_dis_table = cosine_distance(self.testdata, self.traindata)
        self.test_dis_index_table = np.argsort(test_dis_table, axis=1)
    
    # using cosine distance
    def getNeighbours(self, testinstance, k, train = False):
        if train == True:
            return self.train_dis_index_table[testinstance][0:k]
        else:
            return self.test_dis_index_table[testinstance][0:k]
    
    def findLabel(self, kneighbors, k):
        # There are only 2 labels on Face Dataset
        # i.e. 1 or 2
        onesc = 0
        twosc = 0
        for i in range(k):
            l = self.trainlabels[kneighbors[i]]
            if l == 1:
                onesc +=1
            else:
                twosc +=1
        if twosc > onesc:
            return 2
        else:
            return 1
    
    def getAccuracy(self, testlabels, predictions):
        count = 0
        for i in range(len(testlabels)):
            if testlabels[i] == predictions[i]:
                count += 1
        return float(len(testlabels) - count)/float(len(testlabels))
