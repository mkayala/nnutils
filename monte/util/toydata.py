#Copyright (C) 2007-2008 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

from pylab import *
from numpy import *


def Scurve3d(N,Neval,sigma=0.0,h=3.0):
    """ Produce 3d-toydata in the form of an S-shaped set of points. 

        Two datasets are produced: One for training and another, independent 
        one for testing.
    """
    t = pi*(1.5*rand(1,N/2)-1)
    Y = zeros((3,N))
    Y[0,:][newaxis,:] = concatenate((cos(t),-cos(t)),1)
    Y[1,:][newaxis,:] = rand(N)*h
    Y[2,:][newaxis,:] = concatenate((sin(t),2-sin(t)),1)
    Y = Y + randn(3, N) * sigma
    t = concatenate((t,t),1)
  
    teval = pi*(1.5*rand(1,Neval/2)-1)
    Yeval = zeros((3,Neval))
    Yeval[0,:][newaxis,:] = concatenate((cos(teval),-cos(teval)),1)
    Yeval[1,:][newaxis,:] = rand(1, Neval)*h
    Yeval[2,:][newaxis,:] = concatenate((sin(teval),2-sin(teval)),1)
    Yeval = Yeval + randn(3, Neval) * sigma
    teval = concatenate((teval,teval),1)
  
    return Y, Yeval, t, teval

