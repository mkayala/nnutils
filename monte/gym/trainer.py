#Copyright (C) 2007-2008 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

from numpy import mod, inf, double, isnan
from numpy import zeros


def updateparams(model, newparams):
    model.params *= 0.0
    model.params += newparams.copy()


class Trainer(object):
    """ Superclass for trainers. All trainer classes are derived from this. """
    def __init__(self, callback=None, callbackargs=None, callbackiteration=1):
        self.callback = callback
        self.callbackargs = callbackargs
        self.callbackiteration = callbackiteration
        self.iteration = 0
    def step(self, *args):
        self.iteration += 1
        if self.callback and mod(self.iteration,self.callbackiteration)==0:
            self.callback(self.callbackargs,*args)


class Conjugategradients(Trainer):
    """ Fast trainer that makes use of scipy's conjugate gradient optimizer."""
    def __init__(self,model,cgiterations,callback=None,
                                           callbackargs=None,
                                           callbackiteration=1):
        self.model = model
        self.cgiterations = cgiterations
        Trainer.__init__(self,callback,callbackargs,callbackiteration)
    def cost(self,params,*args):
        paramsold = self.model.params.copy()
        updateparams(self.model, params.copy().flatten())
        result = self.model.cost(*args) 
        updateparams(self.model, paramsold.copy())
        return result
    def grad(self,params,*args):
        paramsold = self.model.params.copy()
        updateparams(self.model, params.copy().flatten())
        result = self.model.grad(*args)
        updateparams(self.model, paramsold.copy())
        return result
    def step(self,*args):
        from scipy.optimize import fmin_cg
        updateparams(self.model, fmin_cg(\
                   self.cost,self.model.params.copy(),self.grad,\
                            args=args,maxiter=self.cgiterations,disp=0).copy())
        Trainer.step(self,*args)


class Minimize(Trainer):
    """ Fast trainer that makes use of a Python-version of Carl Rasmussen's
        minimize.m
    """
    def __init__(self,model,maxfuneval,callback=None,
                                       callbackargs=None,
                                       callbackiteration=1):
        self.model = model
        self.maxfuneval = maxfuneval
        Trainer.__init__(self,callback,callbackargs,callbackiteration)
    def cost(self,params,*args):
        paramsold = self.model.params.copy()
        updateparams(self.model, params.copy().flatten())
        result = self.model.cost(*args) 
        updateparams(self.model, paramsold.copy())
        return result
    def grad(self,params,*args):
        paramsold = self.model.params.copy()
        updateparams(self.model, params.copy().flatten())
        result = self.model.grad(*args)
        updateparams(self.model, paramsold.copy())
        return result
    def step(self,*args):
        from minimize import minimize
        updateparams(self.model, minimize(\
                   self.model.params.copy(),self.cost,self.grad,\
                       args=args,maxnumfuneval=self.maxfuneval,
                       verbose=False)[0].copy())
        Trainer.step(self,*args)


class GradientdescentMomentum(Trainer):
    """ Simple online gradient descent with momentum. """
    def __init__(self,model,momentum,stepsize,callback=None,
                                                callbackargs=None,
                                                callbackiteration=1):
        self.model = model                      
        self.momentum = momentum                
        self.stepsize = stepsize
        self.inc = zeros(self.model.params.shape,dtype=float)
        self.oldcost = inf
        self.done = False
        Trainer.__init__(self,callback,callbackargs,callbackiteration)
    def step(self,*args):
        if self.done:
            print 'done. (stepsize <=', str(10**-12), ')'
            return 
        g = self.model.grad(*args)
        self.inc = self.momentum * self.inc - self.stepsize * g
        if isnan(sum(self.inc)): 
            print 'nan!'
            self.inc = zeros(self.inc.shape,dtype=float)
        self.model.params += self.inc
        self.newcost = self.model.cost(*args)
        if self.newcost < self.oldcost:
            self.oldcost = self.newcost
            self.stepsize *= 1.1
        else:
            self.model.params -= self.inc
            self.inc *= 0.0
            #self.inc += (self.inc + self.stepsize * g)/self.momentum
            self.stepsize *= 0.5
            self.newcost = self.oldcost
        if self.stepsize <= 10**-12:
            self.done = True
            #self.newcost = self.oldcost
            #return self.oldcost
        Trainer.step(self,*args)
        #return self.newcost


class OnlinegradientNocost(Trainer):
    """Simple online gradient descent with momentum for model 
       without a cost-function (ie. only the gradient is known)."""
    def __init__(self,model,momentum,stepsize,callback=None,
                                                callbackargs=None,
                                                callbackiteration=1):
        self.model = model
        self.momentum = momentum
        self.stepsize = stepsize
        self.inc = zeros(self.model.params.shape,dtype=float)
        Trainer.__init__(self,callback,callbackargs,callbackiteration)
    def step(self,*args):
        self.inc = \
                self.momentum*self.inc - self.stepsize * self.model.grad(*args)
        if isnan(sum(self.inc)): 
            print 'nan!'
            self.inc = zeros(self.inc.shape,dtype=float)
        self.model.params += self.inc
        Trainer.step(self,*args)


class OnlinegradientNocost(Trainer):
    """Simple online gradient descent with momentum for model 
       without a cost-function (ie. only the gradient is known)."""
    def __init__(self,model,momentum,stepsize,callback=None,
                                                callbackargs=None,
                                                callbackiteration=1):
        self.model = model
        self.momentum = momentum
        self.stepsize = stepsize
        self.inc = zeros(self.model.params.shape,dtype=float)
        Trainer.__init__(self,callback,callbackargs,callbackiteration)
    def step(self,*args):
        self.inc = \
                self.momentum*self.inc - self.stepsize * self.model.grad(*args)
        if isnan(sum(self.inc)): 
            print 'nan!'
            self.inc = zeros(self.inc.shape,dtype=float)
        self.model.params += self.inc
        Trainer.step(self,*args)


class Bolddriver(Trainer):
    """Simple gradient descent with adaptive stepsize. 
       Use for batch-settings."""
    def __init__(self, model, verbose=False, callback=None,
                                               callbackargs=None,
                                               callbackiteration=1):
        self.verbose = verbose 
        self.model = model   
        self.stepsize = 0.1  
        self.oldcost = inf
        self.firstcall = True
        Trainer.__init__(self,callback,callbackargs,callbackiteration)
    def step(self,*args):
        if self.stepsize<10**-12: 
            print "stepsize < 10**-12: exiting" 
            return
        if self.firstcall:
            self.firstcall = False
            self.oldcost = self.model.cost(*args)
            if self.verbose:
                print "initial cost: %f " % self.oldcost
        g = self.model.grad(*args)
        self.model.params -= self.stepsize * g
        self.newcost = self.model.cost(*args)
        if self.newcost <= self.oldcost:
            if self.verbose:
                print "cost: %f " % self.newcost
                print "increasing step-size to %f" % self.stepsize
            self.oldcost = self.newcost
            self.stepsize = self.stepsize * 1.1
        else:
            if self.verbose:
                print "cost: %f larger than best cost %f" % \
                                                  (self.newcost, self.oldcost)
                print "decreasing step-size to %f" % self.stepsize
            self.model.params += self.stepsize * g
            self.newcost = self.oldcost
            self.stepsize = self.stepsize * 0.5
        Trainer.step(self,*args)

