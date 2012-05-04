#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

"""K-means clustering.

   The module contains a single class that performs clustering with the 
   k-means algorithm.
"""

from numpy import zeros, newaxis, inf, sum
from pylab import randn


class Kmeans(object):
    """ K-means clustering. 
  
        Two methods: Use learn to train the model, use assigntoclusters to 
        apply it.
    """

    def __init__(self,k,d):
        """ k: number of clusters, d: input space dimensionality"""
        self.k = k
        self.d = d
        self.tol = 10**-6
        self.codebook = randn(self.d,self.k)

    def learn(self,data,numiter):
        """Train for specified number of iterations on the provided data."""
        numcases = data.shape[1]
        self.codebook = data.mean(1)[:,newaxis] + randn(self.d,self.k)
        lasterr = inf
        for iter in range(numiter):
            #for each datacase determine closest codebook vector:
            assignments = self.assigntoclusters(data)
            #check current reconstruction error:
            err = sum(sum((data-self.codebook[:,assignments])**2))/numcases
            print 'err: ', str(err)
            if lasterr - err < self.tol: 
                break
            lasterr = err
            #adapt codebook:
            for j in range(self.k):
                if sum(assignments==j) != 0:
                    self.codebook[:,j] = \
                             sum(data[:,assignments==j],1)/sum(assignments==j)

    def assigntoclusters(self,data):
        """ For provided data, return assingments to clusters."""
        numcases = data.shape[1]
        assignments = zeros(numcases, dtype=int)
        numcases = data.shape[1]
        for i in range(numcases):
            mindist = inf
            for j in range(self.k):
                dist = sum((data[:,i]-self.codebook[:,j])**2)
                if dist < mindist:
                    assignments[i] = j
                    mindist = dist
        return assignments

