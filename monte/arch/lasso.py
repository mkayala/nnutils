#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

from numpy import dot, newaxis
from pylab import randn


class Pick(object):
    def __init__(self,i):
        self.i = i
    def __call__(self,x,inp,out): return x[self.i]


class Lasso(object):
    """ Tibshirani's Lasso (L1-regularized sparse regression). 
    
    Use the 'train'-method to train the model (via constrained optimization,
    using scipy's fmin_cobyla). The method expects an input-array, an output 
    array and a tuning parameter t that controls the sparseness. After 
    training, use 'getparams' to get the model parameters for inspection 
    ('getparams' returns a tuple containing a coefficient vector and an 
    offset), and use 'apply' to apply the model to test data.
    """
    def __init__(self,numdims):
        self.numdims = numdims
        self.params = 0.001 * randn(numdims*2)
        self.wp = self.params[:numdims]
        self.wm = self.params[numdims:]
        self.b = 0.0
        self.t = 1.0
    def objective(self,params,inp,out):
        wp = params[:self.numdims]
        wm = params[self.numdims:]
        return sum(sum((out - (dot((wp - wm) , inp) + self.b))**2))
    def grad(self,params,inp,out):
        wp = params[:self.numdims]
        wm = params[self.numdims:]
        b  = params[-1]
        gp = 2 * sum((out-dot((wp - wm),inp) + b)[newaxis,:] * inp,1)
        #gm = -2 * sum((out-dot((wp - wm),inp) + b)[newaxis,:] * inp,1)
        gm = - gp.copy()
        return vstack((gp,gm))
    def updateparams(self,newparams):
        self.params *= 0.0
        self.params += newparams.copy()
    def train(self,inp,out,t):
        from scipy.optimize import fmin_cobyla
        self.t = t
        self.b = out.mean()
        constr = [lambda p, inp, out: self.t - sum(p)]
        for i in range(self.params.shape[0]):
            constr.append(Pick(i))
        self.updateparams(fmin_cobyla(self.objective,self.params.copy(),\
                                               constr,(inp,out),maxfun=100000))
    def apply(self,inp):
        return dot((self.wp - self.wm) , inp) + self.b
    def getparams(self):
        return self.wp - self.wm, self.b

