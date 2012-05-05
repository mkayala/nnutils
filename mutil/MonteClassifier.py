#!/usr/bin/env python
# encoding: utf-8
"""
Class to Train a simple monte classifier
working directly with arrays

MonteClassifier.py

Created by Matt Kayala on 2010-05-06.
"""
import sys
import os
from optparse import OptionParser;
import pprint;

from numpy import arange, multiply, log, newaxis, zeros, min, max, any, isnan, where, mean, concatenate;
from numpy.random import shuffle;

import MonteArchModel;
from MonteTrainers import loadMonteTrainer;
from MonteNeuralNetLayer import MonteNeuralNetClassifier;

from Util import log as myLog;
from Const import OFFSET_EPSILON

class MonteClassifier:
    """Class to train or predict with a basic classifier"""
    def __init__(self, archModel=None, dataArr=None, targetArr=None, callback=None, chunklog=False, epochlog=True):
        """Constructor"""
        self.archModel = archModel;
        self.dataArr = dataArr;
        self.targetArr = targetArr;
        
        self.layerModel = None;
        self.trainer = None;
        
        # Callback function after each epoch.  Sends self
        self.callback = callback;

        ## Control the logging
        self.chunklog = chunklog
        self.epochlog = epochlog
    
    
    def setupModels(self):
        """Build the basic trainer setup - based on the ArchModel"""
        # Meat
        self.params = self.archModel.params;
        self.layerModel = MonteNeuralNetClassifier(self.archModel.numfeats, self.archModel.numhidden, self.params);
        self.trainer = loadMonteTrainer(self, self.archModel);
        
        # Minor details
        self.batch = self.archModel.batch;
        self.onlineChunkSize = self.archModel.onlineChunkSize;
        self.gradientChunkSize = self.archModel.gradientChunkSize;
        self.numEpochs = self.archModel.numEpochs;
        self.l2decay = self.archModel.l2decay;
        self.costEpsilon = self.archModel.costEpsilon;
    
    
    def train(self):
        """Method to run through all the data and train away"""
        self.costTrajectory = [];
        
        # Set some things up if doing online
        c1Idx = None;
        c0Idx = None;
        if not self.batch:
            shuffIdx = arange(0, self.dataArr.shape[0]);
            c1Idx = shuffIdx[self.targetArr == 1];
            c0Idx = shuffIdx[self.targetArr == 0];
            numOnlineRuns = int(self.dataArr.shape[0]/self.onlineChunkSize) + 1;
            c1Step = int(len(c1Idx) / numOnlineRuns) + 1;
            c0Step = int(len(c0Idx) / numOnlineRuns) + 1;
        
        try:
            for iEpoch in range(self.numEpochs):
                if self.batch:
                    self.trainer.step(self.dataArr, self.targetArr);
                else:
                    # Want to balance each of the chunks used in the online learning.        
                    shuffle(c1Idx);
                    shuffle(c0Idx);
                    
                    # Calculate an amount to adjust the gradient by because of online 
                    onlineAdjustmentFactor = None; #float(self.onlineChunkSize) / float(self.dataArr.shape[0]);
                    
                    for iOnlineRun in range(numOnlineRuns):
                        c1RowStart = iOnlineRun * c1Step;
                        c1RowEnd = c1RowStart + c1Step;
                        c0RowStart = iOnlineRun * c0Step;
                        c0RowEnd = c0RowStart + c0Step;
                        theInds = concatenate((c1Idx[c1RowStart:c1RowEnd], c0Idx[c0RowStart:c0RowEnd]))
                        subData = self.dataArr[theInds, :];
                        subTargets = self.targetArr[theInds];
                        self.trainer.step(subData, subTargets, onlineAdjustmentFactor);

                if self.epochlog:
                    myLog.debug('About to call cost in postEpoch call')
                currCost = self.cost(self.dataArr, self.targetArr, dolog=self.epochlog);

                if self.epochlog:
                    myLog.info('Epoch %d, curr Cost : %f' % (iEpoch, currCost));
                self.costTrajectory.append(currCost);
                
                # Test for convergence
                if len(self.costTrajectory) > 1:
                    if abs(self.costTrajectory[-1] - self.costTrajectory[-2]) < self.costEpsilon:
                        myLog.critical('Convergence after Epoch %d!!' % iEpoch);
                        return self.costTrajectory;
                
                if self.callback is not None:
                    self.callback(self);
            
            myLog.critical('Never completely converged after %d epochs!' % self.numEpochs);
        except KeyboardInterrupt, e:
            myLog.critical('Interrupted with Keyboard after %d epochs, stopping here, currCost = %f' % (iEpoch, self.costTrajectory[-1]))
            return self.costTrajectory;
        return self.costTrajectory;
    
    
    def cost(self, dataArr, targetArr, onlineAdjustmentFactor=None, dolog=False):
        """Evaluate the error function.
        
        Even in batch mode, we iterate over the data in smaller chunks.
        """
        theCost = 0;
        for subData, subTarg in self.gradChunkDataIterator(dataArr, targetArr):            
            outputs = self.layerModel.fprop(subData);
            outputs = where(outputs < OFFSET_EPSILON, OFFSET_EPSILON, outputs);
            outputs = where(outputs > 1-OFFSET_EPSILON, 1-OFFSET_EPSILON, outputs);
            
            # Cross-Entropy
            error = multiply(subTarg, log(outputs)) + multiply(1 - subTarg, log(1-outputs));
            newCostContrib = error.sum();
            theCost -= newCostContrib;
        
        decayContribution = self.l2decay * (self.params**2).sum();
        
        if onlineAdjustmentFactor is not None:
            # Basically a way to make the cost at each small step be a little smaller (cause 
            # we end up taking more gradient steps)
            decayContribution *= onlineAdjustmentFactor;

        if dolog:
            myLog.debug('decayContribution : %.4f, cost : %.4f' % (decayContribution, theCost));
        theCost += decayContribution;
        
        return theCost;
    
    
    def gradChunkDataIterator(self, dataArr, targetArr):
        """Convenience generator to return small chunks of data to minimize mem usage at a time.
        yields subData, subTarg
        
        Governed by the setting of self.gradientChunkSize
        """
        for rowStart in range(0, dataArr.shape[0], self.gradientChunkSize):
            rowEnd = rowStart + self.gradientChunkSize;
            yield dataArr[rowStart:rowEnd, :].T, targetArr[rowStart:rowEnd].T;
    
    
    def grad(self, dataArr, targetArr, onlineAdjustmentFactor=None):
        """Evaluate the gradient of the error function wrt the params"""
        currGrad = zeros(self.params.shape);

        meanDOut = [];
        minDOut = [];
        maxDOut = [];
        for subData, subTarg in self.gradChunkDataIterator(dataArr, targetArr):
            actual_out = self.layerModel.fprop(subData);
            d_outputs = actual_out - subTarg;
            meanDOut.append(mean(abs(d_outputs)));
            minDOut.append(min(d_outputs));
            maxDOut.append(max(d_outputs));
            self.layerModel.bprop(d_outputs, subData);
            currGradContrib = self.layerModel.grad(d_outputs, subData)
            currGradContrib = currGradContrib.sum(1);
            currGrad += currGradContrib;

        decayContribution = 2 * self.l2decay * self.params;
        
        if onlineAdjustmentFactor is not None:
            decayContribution *= onlineAdjustmentFactor;
        
        currGrad += decayContribution;
        if self.chunklog:
            myLog.debug('||currGrad||^1 : %.4f, ||decayContribution|| : %.4f, mean(currGrad) : %.4f, max(currGrad) : %.4f' % \
                        (abs(currGrad).sum(), self.l2decay * (self.params**2).sum(), mean(currGrad), max(abs(currGrad))));
        return currGrad;
    
    
    def apply(self, newData):
        """Apply the trained neural net to this new data"""
        outData = zeros(newData.shape[0]);
        for dataChunk, outDataChunk in self.gradChunkDataIterator(newData, outData):
            actOut = self.layerModel.fprop(dataChunk);
            outDataChunk += actOut.flatten();
        return outData.flatten();
    

if __name__ == '__main__':
    print "Not meant to be called from command line directly"
    sys.exit(-1);
