#!/usr/bin/env python
# encoding: utf-8
"""
Class to train and apply neural net models to feature dict models.

MonteFeatDictClassifier.py

Created by Matt Kayala on 2010-06-16.
"""
import sys, os
import pprint;

from numpy import arange, multiply, log, newaxis, zeros, min, max;
from numpy import any, isnan, where, mean, concatenate, array, std;
from numpy.random import shuffle;

import MonteArchModel;
from MonteTrainers import loadMonteTrainer;
from MonteNeuralNetLayer import MonteNeuralNetClassifier;

from Util import log as myLog, accuracy, sigmoid, rmse;
from Const import OFFSET_EPSILON

class MonteFeatDictClassifier:
    """Class to train or predict with a basic classifier, using feature dictionaries"""
    def __init__(self, archModel=None, fDictList=None, targetArr=None, idxArr=None, callback=None,
                 chunklog=True, epochlog=True):
        """Constructor
        
        archModel - is a MonteArchModel object with parameters and machine specifics
        fDictList - is a list of dictionaries.  The keys of the dictionary should refernence columns 
            of an array.  The size can be less than the idxArr or targArr.  
            Though the max(idxArr)==len(fDictList).
        targetArr - is an array of the target values.  len(targetArr) == len(idxArr)
        idxArr - an array of indices referencing the fDictList.  
        """
        self.archModel = archModel;
        self.fDictList = fDictList;
        self.targetArr = targetArr;
        self.idxArr = idxArr;
        
        self.layerModel = None;
        self.trainer = None;
        
        self.dataArr = None;
        
        # Callback function after each epoch.  Sends self
        self.callback = callback;
        
        self.costTrajectory = [];
        
        ## Logging
        self.chunklog = chunklog
        self.epochlog = epochlog

        ## Convergence:
        self.nconvergesteps = 3
        self.checkconverge = False

        self.saveparams = False
        self.paramHistory = []
    
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
        if self.idxArr is not None and len(self.idxArr) > 0:
            self.l2decay /= float(len(self.idxArr))
        self.totalL2Decay = self.archModel.l2decay
        
        ## Sets up when we call something converged.
        ## The sd of costs in the costTrajectory over the past convergeEpochs
        ## must be less than costEpsilon to be considered as converged.
        self.costEpsilon = self.archModel.costEpsilon;
        self.convergeEpochs = self.nconvergesteps;
        
        # Set up the data array
        self.dataArr = zeros((self.gradientChunkSize, self.archModel.numfeats));

        if self.saveparams:
            self.archModel.paramHistory = paramHistory
    
    def train(self):
        """Method to run through all the data and train away"""
        self.costTrajectory = [];
        
        # Set some things up if doing online
        c1Idx = None;
        c0Idx = None;
        if not self.batch:
            shuffIdx = arange(0, len(self.idxArr));
            c1Idx = shuffIdx[self.targetArr == 1];
            c0Idx = shuffIdx[self.targetArr == 0];
            #myLog.debug('len(c1Idx) : %d, len(c0Idx) : %d' % (len(c1Idx), len(c0Idx)));
            numOnlineRuns = int(len(self.idxArr)/float(self.onlineChunkSize)) + 1;
            c1Step = int(len(c1Idx) / numOnlineRuns) + 1;
            c0Step = int(len(c0Idx) / numOnlineRuns) + 1;
        
        try:
            for iEpoch in range(self.numEpochs):
                if self.batch:
                    self.trainer.step(self.fDictList, self.targetArr, self.idxArr);
                else:
                    # Want to balance each of the chunks used in the online learning.        
                    shuffle(c1Idx);
                    shuffle(c0Idx);
                    
                    
                    for iOnlineRun in range(numOnlineRuns):
                        c1RowStart = iOnlineRun * c1Step;
                        c1RowEnd = c1RowStart + c1Step;
                        c0RowStart = iOnlineRun * c0Step;
                        c0RowEnd = c0RowStart + c0Step;
                        theInds = concatenate((c1Idx[c1RowStart:c1RowEnd], c0Idx[c0RowStart:c0RowEnd]))
                        shuffle(theInds)
                        #myLog.debug('minshuffidx : %d, maxshuffidx: %d' % (min(theInds), max(theInds)))
                        subTargets = self.targetArr[theInds];
                        subIdx = self.idxArr[theInds];
                        self.trainer.step(self.fDictList, subTargets, subIdx);

                if self.epochlog:
                    myLog.debug('About to call cost in postEpoch call')
                self.postEpochCall(iEpoch);
                
                # Test for convergence
                if self.checkconverge and len(self.costTrajectory) > self.convergeEpochs + 1:
                    if std(self.costTrajectory[-self.convergeEpochs:]) < self.costEpsilon:
                        myLog.critical('Convergence after Epoch %d!!' % iEpoch);
                        return self.costTrajectory;
                
                if self.callback is not None:
                    self.callback(self);
            
            myLog.critical('Never completely converged after %d epochs!' % self.numEpochs);
        except KeyboardInterrupt, e:
            myLog.critical('Interrupted with Keyboard after %d epochs, stopping here, currCost = %f' % (iEpoch, self.costTrajectory[-1]))
            return self.costTrajectory;
        return self.costTrajectory;
    
    
    def postEpochCall(self, epoch):
        """Convenience to run some stats at the end of each epoch."""
        outputs = self.apply(self.fDictList);
        outputs = outputs[self.idxArr];
        outputs = where(outputs < OFFSET_EPSILON, OFFSET_EPSILON, outputs);
        outputs = where(outputs > 1-OFFSET_EPSILON, 1-OFFSET_EPSILON, outputs);
        
        error = multiply(self.targetArr, log(outputs)) + multiply(1 - self.targetArr, log(1-outputs));
        currCost = -error.sum();
        decayContrib = self.totalL2Decay * (self.params**2).sum();
        theAcc = accuracy(outputs, self.targetArr);
        theRMSE = rmse(outputs, self.targetArr);

        if self.epochlog:
            myLog.info('Epoch %d, curr Cost: %f, decayCont: %f ' % (epoch, currCost, decayContrib));
            myLog.info('Epoch %d, theAcc: %f, theRMSE: %f' % (epoch, theAcc, theRMSE));
        self.costTrajectory.append(currCost);
    
    
    def cost(self, fDictList, targetArr, idxArr):
        """Evaluate the error function.
        
        Even in batch mode, we iterate over the data in smaller chunks.
        """
        theCost = 0;
        for subData, subTarg in self.gradChunkDataIterator(fDictList, targetArr, idxArr):            
            outputs = self.layerModel.fprop(subData);
            outputs = where(outputs < OFFSET_EPSILON, OFFSET_EPSILON, outputs);
            outputs = where(outputs > 1-OFFSET_EPSILON, 1-OFFSET_EPSILON, outputs);
            
            # Cross-Entropy
            error = multiply(subTarg, log(outputs)) + multiply(1 - subTarg, log(1-outputs));
            newCostContrib = error.sum();
            theCost -= newCostContrib;
        
        decayContribution = self.l2decay * (self.params**2).sum() * len(idxArr);
        
        if self.chunklog:
            myLog.debug('decayContribution : %.4f, cost : %.4f' % (decayContribution, theCost));
        theCost += decayContribution;
        
        return theCost;
    
    
    def gradChunkDataIterator(self, fDictList, targetArr, idxArr):
        """Convenience generator to return small chunks of data to minimize mem usage at a time.
        yields subData, subTarg
        
        Converts the fDictList into the dataArr.  (Assumes that the keys of the dictionary
        are the columns);
        """
        for rowStart in range(0, len(idxArr), self.gradientChunkSize):
            self.dataArr *= 0.0;
            rowEnd = rowStart + self.gradientChunkSize;
            subIdxArr = idxArr[rowStart:rowEnd];
            subTargArr = targetArr[rowStart:rowEnd];
            for iRow, fDictIdx in enumerate(subIdxArr):
                for colIdx, val in fDictList[fDictIdx].iteritems():
                    try:
                        self.dataArr[iRow, colIdx] = val;
                    except Exception, e:
                        myLog.critical('iRow: %s, colIdx: %s, val: %s' % (pprint.pformat(iRow), pprint.pformat(colIdx), pprint.pformat(val)));
                        raise e;
            numRows = len(subIdxArr);
            yield self.dataArr[:numRows, :].T, subTargArr[:numRows].T;
    
    
    def grad(self, fDictList, targetArr, idxArr):
        """Evaluate the gradient of the error function wrt the params"""
        currGrad = zeros(self.params.shape);

        meanDOut = [];
        minDOut = [];
        maxDOut = [];
        for subData, subTarg in self.gradChunkDataIterator(fDictList, targetArr, idxArr):
            actual_out = self.layerModel.fprop(subData);
            d_outputs = actual_out - subTarg;
            meanDOut.append(mean(abs(d_outputs)));
            minDOut.append(min(d_outputs));
            maxDOut.append(max(d_outputs));
            self.layerModel.bprop(d_outputs, subData);
            currGradContrib = self.layerModel.grad(d_outputs, subData)
            currGradContrib = currGradContrib.sum(1);

            currGrad += currGradContrib;

        decayContribution = 2 * self.l2decay * self.params * len(idxArr);
                
        currGrad += decayContribution;
        if self.chunklog:
            myLog.debug('mean(abs(d_outputs)) : %.4f, min(d_outputs): %.4f, max(d_outputs) : %.4f' % \
                        (mean(meanDOut), min(minDOut), max(maxDOut)))
            myLog.debug('||currGrad||^1 : %.4f, ||decayContribution|| : %.4f, mean(currGrad) : %.4f, max(currGrad) : %.4f' % \
                        (abs(currGrad).sum(), self.l2decay * (self.params**2).sum() * len(idxArr), mean(currGrad), max(abs(currGrad))));
            myLog.debug('max(currGrad) : %.4f, min(currGrad) : %.4f' % (max(currGrad), min(currGrad)))
            myLog.debug('max(params) : %.4f, min(params) : %.4f' % (max(self.params), min(self.params)))
            myLog.debug('max(decayContribution) : %.4f, min(decayContribution) : %.4f' % (max(decayContribution),
                                                                                          min(decayContribution)))
            myLog.debug('len(idxArr) = %d' % len(idxArr))

        return currGrad;
    
    
    def apply(self, newData):
        """Apply the trained neural net to this new data"""
        outData = zeros(len(newData));
        idxArr = array(range(len(newData)));
        for dataChunk, outDataChunk in self.gradChunkDataIterator(newData, outData, idxArr):
            actOut = self.layerModel.fprop(dataChunk);
            outDataChunk += actOut.flatten();
        return outData.flatten();
    
if __name__ == '__main__':
    print "Not meant to be called from command line directly"
    sys.exit(-1);
