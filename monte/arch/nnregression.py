#Copyright (C) 2007-2008 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

""" Nearest neighbor regression."""

from numpy import dot, newaxis, argsort, sum

def nnregression(traininputs, trainoutputs, testinputs, dist=None):
    """ Nearest neighbor regression.
  
        traininputs: Float array of training inputs (columnwise).
        trainoutputs: Float array of training outputs (columnwise).
        testinputs:  Float array of test inputs (columnwise).
        dist: The distance function to use. If dist in None (default) Euclidean 
              distance is used.
  
        Output: Nearest neighbor regression applied to the testinputs. 
    """
    if len(traininputs.shape) < 2:
        traininputs = traininputs[newaxis, :]
    if len(trainoutputs.shape) < 2:
        trainoutputs = trainoutputs[newaxis, :]
    if len(testinputs.shape) < 2:
        testinputs = testinputs[newaxis, :]
    if dist is None:  #use Euclidean distance, if no other provided
        D = sum(traininputs**2,0)[:,newaxis]+sum(testinputs**2,0)[newaxis,:]\
            -2*dot(traininputs.T,testinputs)
    else:
        D = dist(traininputs,testinputs)
    return trainoutputs[:, argsort(D,0)[0,:]]

