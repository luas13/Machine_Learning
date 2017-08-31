# This piece of software is bound by The MIT License (MIT)
# Copyright (c) 2017 Abhishek Kumar
# Code written by : Abhishek Kumar
# Email ID : akuma151@asu.edu

from kNearestNeighbour import kNN
import scipy.io
import matplotlib.pyplot as plt
import timeit


if __name__ == '__main__':
    mat = scipy.io.loadmat('faces.mat')
    k_array = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    knn = kNN(mat)

    start = timeit.default_timer()

    # training phase
    # print '### TRAINING PHASE ###'
    train_error = {}
    for k in k_array:
        predictions = []
        counter = 0
        for trainindex in range(mat['traindata'].shape[0]):
            kneighbours = knn.getNeighbours(trainindex, k, train = True)
            label = knn.findLabel(kneighbours, k)
            predictions.append(label)
            counter += 1
        error = knn.getAccuracy(mat['trainlabels'], predictions)
        train_error[k] = error

    # testing phase
    # print '### TESTING PHASE ###'
    test_error = {}
    for k in k_array:
        predictions = []
        counter = 0
        for testinstance in range(mat['testdata'].shape[0]):
            kneighbours = knn.getNeighbours(testinstance, k)
            label = knn.findLabel(kneighbours, k)
            predictions.append(label)
            counter +=1
        error = knn.getAccuracy(mat['testlabels'], predictions)
        test_error[k] = error
    # print 'k-value : error'
    # print test_error
    stop = timeit.default_timer()

    print 'Training and testing time =', stop - start

    train_lists = sorted(train_error.items())
    k, e = zip(*train_lists)
    plt.plot(k, e, 'r-o', label='train error rate')

    test_lists = sorted(test_error.items())
    k, e = zip(*test_lists)
    plt.plot(k, e, 'g-o', label='test error rate')

    plt.legend(loc=5)

    plt.title("Training and Testing error rate w.r.t. k", weight='bold')
    plt.xlabel("k (no. of neighbours)", weight='bold')
    plt.ylabel("Error rate", weight='bold')
    plt.show()
