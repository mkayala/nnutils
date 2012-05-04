#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

""" k-nearest neighbor classification.

The two functions 'classify' and 'error' classify a test-set using k-nearest
neighbors on a given training set, or calculate the resulting error-rates, 
respectively.
"""

from numpy import zeros, ones, iterable, dot, newaxis, histogram, argsort, \
                  size, double, sum
from pylab import randn, find


def classify(traininputs, trainlabels, testinputs, K, dist=None):
    """ Classification with k-nearest neighbors.
  
        Labels are encoded as integers, with value 0 encoding the first class.
        The input K can be a single integer specifying the 'k' (# of neighbors) 
        to use, or it can be a sequence of several k's, in which case the 
        classification results for each one of them are returned.
  
        traininputs: Float array of training inputs (columnwise).
        trainlabels: Integer array of training labels (can be either rank-1 
                     or rank-2. If rank-2, labels are assumed to be arranged 
                     along the second rank ('columnwise'). In other words, 
                     if the rank is 2, the first rank has dimension 1.
        testinputs:  Float array of test inputs (columnwise).
        K: Either a single integer, specifying the number of neighbors to use,
           or a sequence of integers. In the latter case, results for each 
           are returned.
        dist: The distance function to use. If dist in None (default) Euclidean 
              distance is used.
  
        Output: Array of size (number of elements in K, number of testcases). 
                Each row contains the classification results (as integers) on 
                the test-inputs for one element in K.
    """
    if not iterable(K):
        K = [K]
    if len(trainlabels.shape)==2:
        trainlabels = trainlabels.flatten()
    assert trainlabels.min() == 0
    numclasses = max(trainlabels) + 1
    ntr = traininputs.shape[1]
    nte = testinputs.shape[1]
    globalhist = histogram(trainlabels,bins=numclasses, range=[0,numclasses])[0]
  
    if dist is None:  #use Euclidean distance, if no other provided
        tr2 = sum(traininputs**2,0)
        te2 = sum(testinputs**2,0)
        D = tr2[:,newaxis] + te2[newaxis,:] - 2*dot(traininputs.T,testinputs)
    else:
        D = dist(traininputs,testinputs)
    ind = argsort(D,0)
    testlabels = zeros((len(K),nte),dtype=int)
    for j in range(len(K)):
        for i in range(nte):
            labels = trainlabels[ind[:K[j],i]]
            h = histogram(labels, bins=numclasses, range=[0,numclasses])[0]
            c = find(h==h.max())
            if size(c)>1:  #if ties, back off to 1-nn
                c = trainlabels[ind[0,i]]
            testlabels[j,i] = c
    return testlabels


def error(traininputs, trainlabels, testinputs, testlabels, K, dist=None):
    """ Compute the classification error for k-nearest neighbors.
  
        Labels are encoded as integers, with value 0 encoding the first class.
        The input K can be a single integer specifying the 'k' (# of neighbors) 
        to use, or it can be a sequence of several k's, in which case the 
        results for each one of them are returned.
  
        traininputs: Float array containing the training inputs (columnwise).
        trainlabels: Integer array of training labels (can be either rank-1 
                     or rank-2. If rank-2, labels are assumed to be arranged 
                     along the second rank ('columnwise'). In other words, 
                     if the rank is 2, the first rank has dimension 1.
        testinputs:  Float array of test inputs (columnwise).
        testlabels:  Integer array of correct labels on the test-set (can be 
                     either rank-1 or rank-2. If rank-2, labels are assumed to 
                     be arranged along the second rank ('columnwise'). (In 
                     other words, if the rank is 2, the first rank has 
                     dimension 1).
        K: Either a single integer, specifying the number of neighbors to use,
           or a sequence of integers. In the latter case, results for each 
           are returned.
        dist: The distance function to use. If dist in None (default) Euclidean 
              distance is used.
        Output: Tuple, whos first component contains the error rates for each 
                element in K, and whos second component contains the number 
                of correctly classified cases (also for each element in K).
    """
    if not iterable(K):
        print "not iterable"
        K = [K]
    answers = classify(traininputs, trainlabels, testinputs, K, dist)
    errs = zeros(len(K),dtype=float)
    numcorrect = zeros((len(K)),dtype=int)
    for kk in range(len(K)):
        numcorrect[kk] = sum(answers[kk]==testlabels)
        errs[kk] = 1-double(numcorrect[kk])/double(testinputs.shape[1])
    return errs, numcorrect


if __name__ == '__main__':
    #Knn-classification on toy-data:
    from numpy import hstack
    traininputs = hstack((randn(2,100)-1.0,randn(2,100)+1.0))
    trainlabels = hstack((ones(100)*0,ones(100)*1))
    testinputs = hstack((randn(2,100)-1.0,randn(2,100)+1.0))
    testlabels = hstack((ones(100)*0,ones(100)*1)).astype(int)
    answers = classify(traininputs,trainlabels,testinputs,[1,2,3])
    errrate, numcorrect = \
            error(traininputs,trainlabels,testinputs,testlabels,[1,2,3])
    print 'testlabels:\n', testlabels
    print 'answers for k=1,2,3:\n', answers
    print "num correct:", str(numcorrect)
    print "error rate:", str(errrate)

