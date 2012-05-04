#!/usr/bin/env python
# encoding: utf-8
"""
Simple classes to generalize some monte options.

MonteNeuralNetLayer.py

Created by Matt Kayala on 2010-05-06.
"""

import sys
import os
from numpy import zeros, asmatrix, concatenate, outer, multiply, prod, sum, exp;

from nnutils.monte.bp.neuralnet import SigmoidLinear, Linearlayer;

from Util import log, sigmoid, identity;

class MonteNeuralNetClassifier:
    """Simple class to make a general 0 or 1 layer neural net 
    
    By default, this includes a single sigmoid output node.
    With the constuctor param sigmoidOut=False, you can make this a regressor.
    """
    @staticmethod
    def numparams(numin, numhidden):
        """Compute number of params needed"""
        if numhidden == 0:
            return Linearlayer.numparams(numin, 1)
        else:
            return SigmoidLinear.numparams(numin, numhidden, 1);
    
    def __init__(self, numin, numhidden, params, sigmoidOut=True):
        """Setup some """
        self.numin = numin;
        self.numhidden = numhidden;
        self.numout = 1;
        self.params = params;
        
        self.bplayer = None;
        if self.numhidden == 0:
            self.bplayer = Linearlayer(self.numin, self.numout, self.params);
        else:
            self.bplayer = SigmoidLinear(self.numin, self.numhidden, self.numout, self.params);
        
        self.sigmoidOut = sigmoidOut;
        self.fpropFinalizeFunc = sigmoid;
        if not self.sigmoidOut:
            self.fpropFinalizeFunc = identity;
      
    def fprop(self, input):
        """Feed forward"""
        self.activities = self.bplayer.fprop(input);
        return self.fpropFinalizeFunc(self.activities);
    
    def bprop(self, d_output, input):
        """Note it is the onus of the caller to correctly pass the correct initial error term back.
        
        Back propagate.  Note Bishop Formula 5.54, that for output layer, do not mult by the sigmoid deriv
        if using the canonical link function with cross-entropy error.
        """
        return self.bplayer.bprop(d_output, input);
    
    
    def grad(self, d_output, input):
        """Calculate the gradient"""
        return self.bplayer.grad(d_output, input);

def main(argv):
    pass;

if __name__ == '__main__':
    sys.exit(main(sys.argv));
