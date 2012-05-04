#!/usr/bin/env python
# encoding: utf-8
"""
Module with some different Trainer classes that can be used in the MonteOnlineQualLearner 
code.


MonteTrainers.py

Created by Matt Kayala on 2010-01-04.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.
"""


import sys
import os
from pprint import pformat;

from Util import log, sigmoid;
from Const import EPSILON;

#Using the monte library.. http://montepython.sourceforge.net/
from monte.gym.trainer import Trainer, updateparams;

from numpy import min, max, mean;

from scipy import exp, zeros, shape, randn, newaxis, sqrt, ones;
from scipy import mod, inf, double, isnan, where;
from scipy.optimize import fmin_bfgs, fmin_cg;

class BfgsTrainer(Trainer):
    """Instance of Trainer that uses the BFGS optimization method.
    
    http://tinyurl.com/yhdcv3d"""
    def __init__(self,model,optiterations=10,callback=None,
                                           callbackargs=None,
                                           callbackiteration=1):
        self.model = model
        self.optiterations = optiterations
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
        updateparams(self.model, fmin_bfgs(\
                   self.cost,self.model.params.copy(),self.grad,\
                            args=args,maxiter=self.optiterations,disp=1).copy())
        Trainer.step(self,*args)



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
        updateparams(self.model, fmin_cg(\
                   self.cost,self.model.params.copy(),self.grad,\
                            args=args,maxiter=self.cgiterations,disp=1).copy())
        Trainer.step(self,*args)



class OnlinegradientNocostMomentum(Trainer):
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


class SimpleGradientDescent(Trainer):
    """Simple gradient descent with no adaptive stepsize. """
    def __init__(self, model, stepsize=0.1, verbose=False, callback=None,
                                               callbackargs=None,
                                               callbackiteration=1):
        self.verbose = verbose 
        self.model = model   
        self.stepsize = stepsize;
        Trainer.__init__(self,callback,callbackargs,callbackiteration)
    def step(self,*args):
        g = self.model.grad(*args)
        self.model.params -= self.stepsize * g
        Trainer.step(self,*args)


class AdaptiveLocalStepGradDescent(Trainer):
    """Class to encapsulate the idea of local learning rates that adapt over time for online learning.
    Note, this trainer does not make sense for batch learning.  Should use BoldDriver or something 
    else for that.
    
    http://www.csi.ucd.ie/staff/fcummins/NNcourse/precond.html
    http://www.csi.ucd.ie/staff/fcummins/NNcourse/momrate.html
    
    Three main ideas:  
        1. set the initial global stepsize based on the 1/sqrt(chunksize)
        2. adjust the initial learning rates to be different for diff params based on the arch of the 
            model.  mu_i = mu/(|A_i| \sqrt(v_i)) where |A_i| is number of links into node i.  
            and v_i = 1/|A_i| * sum (v_{parents_i}) with v_i = 1/|A_i| for top node.
        3. Finally, update the learning rate at each step based on overlap with a previous gradient
            and normalized exponential averages.
            
    """
    def __init__(self, model, onlineChunkSize=10000, initialMu=1, exponentAvgM=0.5, qLearningRate=0.90,
                numfeats=1, numhidden=0, numout=1, 
                verbose=False, callback=None, callbackargs=None, callbackiteration=1):
        """Set up the initial stuff"""
        self.verbose = verbose;
        self.model = model;
        self.exponentAvgM = exponentAvgM;
        self.qLearningRate = qLearningRate;
        
        self.muVect = self.setupMu(onlineChunkSize, initialMu, self.model.params.shape, numfeats, numhidden, numout)
        
        self.expAvgGrad = zeros(self.muVect.shape);
        self.sqExpAvgGrad = ones(self.muVect.shape);
        
        Trainer.__init__(self, callback, callbackargs, callbackiteration);
    
    
    def setupMu(self, onlineChunkSize, initialMu, paramShape, numfeats, numhidden, numout):
        """Setup the initial mu vector"""
        globalMu = initialMu / sqrt(4.0 * onlineChunkSize);
        muVect = globalMu * ones(paramShape);
        
        if numhidden == 0:
            # Simple logistic regression, adjust by the number of params
            muVect /= sqrt(numfeats + 1);
        else:
            # Have a hidden layer.   
            muVect[:numfeats*numhidden + numhidden] /= sqrt( float(numfeats + 1)/(numhidden + 1));
            muVect[numfeats * numhidden + numhidden:] /= sqrt(numhidden + 1);
            
        #log.debug('Initial muVect: %s' % pformat(muVect));
        return muVect;
    
    
    def step(self, *args):
        """First update the step size, then actually take a step along the gradient."""
        g = self.model.grad(*args);
        
        # Update the weighted Exponential sq avg.
        self.sqExpAvgGrad *= self.exponentAvgM;
        self.sqExpAvgGrad += (1-self.exponentAvgM) * g**2;
        self.sqExpAvgGrad[:] = where(self.sqExpAvgGrad < EPSILON, EPSILON, self.sqExpAvgGrad);
        
        # Uodate the muVect
        possUpdate = 1 + self.qLearningRate * g * self.expAvgGrad / self.sqExpAvgGrad
        self.muVect *= where(possUpdate < 0.1, 0.1, possUpdate);
        
        # Do something to cap the update rate.  This is allowing the step rate to overpower the decay completely
        self.muVect = where(self.muVect > 5.0, 4.0, self.muVect);
        
        # Then update the exponential average
        self.expAvgGrad *= self.exponentAvgM;
        self.expAvgGrad += (1-self.exponentAvgM) * g;
        
        self.model.params -= self.muVect * g
        Trainer.step(self,*args)
    

def loadMonteTrainer(model, monteArchModel):
    """Convenience function to return properly initialized monte trainer"""
    trainer = None;
    if monteArchModel.trainertype == 'conjgrad':
        trainer = Conjugategradients(model, monteArchModel.cgIterations);
    elif monteArchModel.trainertype == 'gdescmom':
        trainer = OnlinegradientNocostMomentum(model, monteArchModel.momentum, monteArchModel.learningrate);
    elif monteArchModel.trainertype == 'gdesc':
        trainer = SimpleGradientDescent(model, monteArchModel.learningrate)
    elif monteArchModel.trainertype == 'bfgs':
        trainer = BfgsTrainer(model, monteArchModel.cgIterations);
    elif monteArchModel.trainertype == 'gdescadapt':
        trainer = AdaptiveLocalStepGradDescent(model, monteArchModel.onlineChunkSize, monteArchModel.learningrate,
                    monteArchModel.exponentAvgM, monteArchModel.qLearningRate, monteArchModel.numfeats,
                    monteArchModel.numhidden, 1)
    else:
        raise Exception('TrainerType is not specified');
    
    if trainer is not None:
        pass
    
    return trainer;

