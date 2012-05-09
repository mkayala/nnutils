#!/usr/bin/env python
# encoding: utf-8
"""
Class to provide framework to Fit parameters to shared weight neural nets with a final 
hardcoded weight sigmoid output layer performs pairwise classification. 

10-13-2010 - Slight changes here to speed up the training.  Basically, for each pair, we don't
look at it twice, but rather save the results from one fprop and bprop to correctly calc the 
gradient in one step for both sides of the pair.

PairMonteFeatDictClassifier.py

Created by Matt Kayala on 2010-08-12.
"""
import sys
import os
import pprint;

from numpy import arange, multiply, log, newaxis, zeros, min, max;
from numpy import any, isnan, where, mean, concatenate, array, ones, zeros;
from numpy.random import shuffle;

import MonteArchModel;
from MonteTrainers import loadMonteTrainer;
from MonteNeuralNetLayer import MonteNeuralNetClassifier;

from Util import log as myLog, sigmoid, accuracy, rmse;
from Const import OFFSET_EPSILON

class PairMonteFeatDictClassifier:
    """Class to fit params for pair wise classification using shared weight neural nets"""
    def __init__(self, archModel=None, fDictList=None, problemArr=None, callback=None, chunklog=False, epochlog=True):
        """Constructor.
        
        archModel - is a MonteArchModel object with parameters and machine specifics
        fDictList - is a list of dictionaries.  The keys of the dictionary should refernence columns 
            of an array.  The size can be less of more than the probArr.  
            Though the max(probArr) < len(fDictList).
        problemArr - is an array of pairs of indices of the fDictList in ordered manner.
            (id0, id1)  means that f(fDictList[id0]) > f(fDictList[id1]) 
        callback - function to call at the end of each epoch.
        """
        self.archModel = archModel;
        self.fDictList = fDictList;
        self.problemArr = problemArr;
        
        # Specific machinery
        self.layerModel = None;
        self.trainer = None;
        
        # To hold the arrays uses to calculate small chunks of the gradient
        self.lDataArr = None;
        self.rDataArr = None;
        
        # Callback function after each epoch.  Sends self
        self.callback = callback;
        
        self.costTrajectory = [];

        ## Logging options
        self.chunklog = chunklog
        self.epochlog = epochlog
    
    def setupModels(self):
        """Build the basic trainer setup - based on the ArchModel"""
        # Meat
        self.params = self.archModel.params;
        # Set this up to be a regressor.
        self.layerModel = MonteNeuralNetClassifier(self.archModel.numfeats,
                                                   self.archModel.numhidden,
                                                   self.params,
                                                   sigmoidOut=False);
        self.trainer = loadMonteTrainer(self, self.archModel);
        
        # Minor details
        self.batch = self.archModel.batch;
        self.onlineChunkSize = self.archModel.onlineChunkSize;
        self.gradientChunkSize = self.archModel.gradientChunkSize;
        self.numEpochs = self.archModel.numEpochs;
        self.l2decay = self.archModel.l2decay;
        if self.fDictList is not None and len(self.fDictList) > 0:
            self.l2decay /= len(self.fDictList)
        self.totalL2Decay = self.archModel.l2decay
        self.costEpsilon = self.archModel.costEpsilon;
        
        # Set up the data array
        self.lDataArr = zeros((self.gradientChunkSize, self.archModel.numfeats));
        self.rDataArr = zeros((self.gradientChunkSize, self.archModel.numfeats));
    
    
    def train(self):
        """Method to run through all the data and train away"""
        # Set some things up if doing online
        # will only calculate the gradient on the left hand side at a time 
        # So make a target array and make sure we look at each ordered pair in both orders
        numOnlineRuns = int((self.problemArr.shape[0])/float(self.onlineChunkSize)) + 1;
        theIdx = arange(0, self.problemArr.shape[0])
        theStep = int(len(theIdx) / numOnlineRuns) + 1;

        ## Check if we have some null rows.  If we do, then stratify the rows trained on in each 
        ## online step
        ## We don't want a complete step of left -1's or right -1's.
        isNullRows = (self.problemArr[:,0] == -1) |  (self.problemArr[:, 1] == -1)
        isNullRows = any(isNullRows)
        if isNullRows:
            myLog.info('Have -1 in problemArr, stratifying the online step data')
            nNullIdx = theIdx[(self.problemArr[:,0] != -1) & (self.problemArr[:,1] != -1)]
            lNullIdx = theIdx[self.problemArr[:,0] == -1]
            rNullIdx = theIdx[self.problemArr[:,1] == -1]

            nNullStep = int(len(nNullIdx)/numOnlineRuns) + 1
            lNullStep = int(len(lNullIdx)/numOnlineRuns) + 1
            rNullStep = int(len(rNullIdx)/numOnlineRuns) + 1
        
        try:
            for iEpoch in range(self.numEpochs):
                if self.batch:
                    self.trainer.step(self.fDictList, self.problemArr);
                else:
                    # Want to balance each of the chunks used in the online learning.
                    if isNullRows:
                        shuffle(nNullIdx);
                        shuffle(lNullIdx)
                        shuffle(rNullIdx)
                    else:
                        shuffle(theIdx);
                                        
                    for iOnlineRun in range(numOnlineRuns):
                        if isNullRows:
                            nNullStart = iOnlineRun * nNullStep
                            nNullEnd = nNullStart + nNullStep
                            lNullStart = iOnlineRun * lNullStep
                            lNullEnd = lNullStart + lNullStep
                            rNullStart = iOnlineRun * rNullStep
                            rNullEnd = rNullStart + rNullStep

                            subProbArr = concatenate((self.problemArr[nNullIdx[nNullStart:nNullEnd], :],
                                                      self.problemArr[lNullIdx[lNullStart:lNullEnd], :],
                                                      self.problemArr[rNullIdx[rNullStart:rNullEnd], :]))
                            
                        else:
                            rowStart = iOnlineRun * theStep;
                            rowEnd = rowStart + theStep;
                            subIdx = theIdx[rowStart:rowEnd];
                            subProbArr = self.problemArr[subIdx, :]
                        self.trainer.step(self.fDictList, subProbArr);
                
                myLog.debug('About to call cost in postEpoch call')
                self.postEpochCall(iEpoch)
                
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
    
    def postEpochCall(self, epoch):
        """Convenience to run some stats at the end of each epoch."""
        outputs = self.apply(self.fDictList);
        lOut = outputs[self.problemArr[:, 0]]
        rOut = outputs[self.problemArr[:, 1]]
        sigOut = sigmoid(lOut - rOut);
        sigOut = where(sigOut < OFFSET_EPSILON, OFFSET_EPSILON, sigOut);
        sigOut = where(sigOut > 1-OFFSET_EPSILON, 1-OFFSET_EPSILON, sigOut);
        # Cross-Entropy
        # NOTE here that all targs are 1.
        error = log(sigOut)
        currCost = -error.sum();
        
        decayContrib = self.totalL2Decay * (self.params**2).sum();
        theAcc = accuracy(sigOut, 1);
        theRMSE = rmse(sigOut, 1);

        if self.epochlog:
            myLog.info('Epoch %d, curr Cost: %f, decayCont: %f ' % (epoch, currCost, decayContrib));
            myLog.info('Epoch %d, theAcc: %f, theRMSE: %f' % (epoch, theAcc, theRMSE));
        self.costTrajectory.append(currCost);
    
    
    def singleSideGradChunkDataIterator(self, fDictList):
        """Convenience to yield out self.gradientChunkSize arrays of the data encoded in fDictList"""
        fDictIdx = arange(0, len(fDictList));
        for rowStart in range(0, len(fDictList), self.gradientChunkSize):
            self.lDataArr *= 0.0;
            rowEnd = rowStart + self.gradientChunkSize;
            subIdxArr = fDictIdx[rowStart:rowEnd]
            for iRow, subIdxVal in enumerate(subIdxArr):
                self.mapFDictToArray(fDictList[subIdxVal], self.lDataArr, iRow)
            
            numRows = subIdxArr.shape[0];
            yield self.lDataArr[:numRows, :].T
    
    def mapFDictToArray(self, fd, arr, iRow):
        """Convenience to map a fd to a particular row in an array (where keys are the columns)"""
        for colIdx, val in fd.iteritems():
            try:
                arr[iRow, colIdx] = val;
            except Exception, e:
                myLog.critical('lData, iRow: %s, colIdx: %s, val: %s' % (pprint.pformat(iRow), pprint.pformat(colIdx), pprint.pformat(val)));
                raise e;
    
    def gradChunkDataIterator(self, fDictList, probArr):
        """Convenience generator to return small chunks of data to minimize mem usage at a time.
        yields subData, subTarg
        
        Converts the fDictList into an lDataArr and rDataArr.  (Assumes that the keys of the dictionary
        are the columns);

        Changed to setup a null version.
        If the id for a given row is -1, then set the corresponding row to be ALL zeros.
        This is used in ranking to impose an order around a classification threshold.  Eg:
        If we had the following data
        grp1 rowid1 class1
        grp1 rowid2 class1
        grp1 rowid3 class0
        grp1 rowid4 class0

        We might have the following problemArr
        rowid1 rowid3
        rowid1 rowid4
        rowid1 -1
        rowid2 rowid3
        rowid2 rowid4
        rowid2 -1
        -1 rowid3
        -1 rowid4

        rowid1 and rowid2 should be greater than both rowid3 and rowid4 AND greater than zero
        rowid3 and rowid4 should be LESS than zero
        """
        for rowStart in range(0, probArr.shape[0], self.gradientChunkSize):
            self.lDataArr *= 0.0;
            self.rDataArr *= 0.0;
            rowEnd = rowStart + self.gradientChunkSize;
            subProbArr = probArr[rowStart:rowEnd, :];
            for iRow, subIdxArr in enumerate(subProbArr):
                lIdx = subIdxArr[0]
                rIdx = subIdxArr[1]
                
                # Map only if id is not -1
                if lIdx == -1:
                    self.lDataArr[iRow, :] *= 0.0
                else:
                    self.mapFDictToArray(fDictList[lIdx], self.lDataArr, iRow)
                if rIdx == -1:
                    self.rDataArr[iRow, :] *= 0.0
                else:
                    self.mapFDictToArray(fDictList[rIdx], self.rDataArr, iRow)
                
            numRows = subProbArr.shape[0];
            yield self.lDataArr[:numRows, :].T, self.rDataArr[:numRows].T, subProbArr;
    
    def cost(self, fDictList, problemArr):
        """Evaluate the error function.
        
        This is based off of a sigmoid over the difference from the left-right outputs.
        Assume that the lArr is always 'preferred' over the rArr.
        """
        theCost = 0;
        for lSubData, rSubData, subProbArr in self.gradChunkDataIterator(fDictList, problemArr):
            #lTargs = ones(problemArr.shape[0])
            # Here we simply need to calc the cost 
            lOut = self.layerModel.fprop(lSubData)
            rOut = self.layerModel.fprop(rSubData)
            
            outputs = sigmoid(lOut - rOut)
            outputs = where(outputs < OFFSET_EPSILON, OFFSET_EPSILON, outputs);
            outputs = where(outputs > 1-OFFSET_EPSILON, 1-OFFSET_EPSILON, outputs);
            # Cross-Entropy
            # NOTE here that all targs are 1.
            error = log(outputs)
            newCostContrib = error.sum();
            theCost -= newCostContrib;
        
        decayContribution = self.l2decay * (self.params**2).sum() * problemArr.shape[0];

        if self.chunklog:
            myLog.debug('decayContribution : %.4f, cost : %.4f' % (decayContribution, theCost));
        theCost += decayContribution;
        
        return theCost;
    
    
    def grad(self, fDictList, problemArr):
        """Evaluate the gradient of the error function wrt the params.
        
        The first error to pass back to the NN will be the 
        (target - sigmoid(lOut - rOut)) * (abs(lOut - rOut))
        
        NOTE:  all targs are 1.  So calc the gradient with l/r then with r/l
        """
        currGrad = zeros(self.params.shape);
        
        meanDOut = [];
        minDOut = [];
        maxDOut = [];
        for lSubData, rSubData, subProbArr in self.gradChunkDataIterator(fDictList, problemArr):
            # First calc the contribution when looking at the left to right
            ## Only include data that comes from non-NULL on the left
            nonLNullRows = subProbArr[:,0]>-1 
            rd = rSubData[:, nonLNullRows]
            ld = lSubData[:, nonLNullRows]
            rOut = self.layerModel.fprop(rd)
            lOut = self.layerModel.fprop(ld)
            sigOut = sigmoid(lOut - rOut);
            absDiffOut = abs(lOut - rOut);

            d_outputs = (sigOut - 1);
            if len(d_outputs) > 0 and d_outputs.shape[1] > 0:
                try:
                    meanDOut.append(mean(abs(d_outputs)));
                    minDOut.append(min(d_outputs));
                    maxDOut.append(max(d_outputs));
                except Exception, e:
                    myLog.error(e)
                    myLog.info('Some error, d_outputs : %s' % pprint.pformat(d_outputs))
                    myLog.info('Some error, len(d_outputs) : %s' % len(d_outputs))
                    raise e
            
                self.layerModel.bprop(d_outputs, ld);
                currGradContrib = self.layerModel.grad(d_outputs, ld)
                currGradContrib = currGradContrib.sum(1);
                currGrad += currGradContrib;
            
            ## Then do the same, but going to the right.
            # need to fprop on the right side again
            # Only use rows that are non-null on the right
            nonRNullRows = subProbArr[:,1]>-1 
            rd = rSubData[:, nonRNullRows]
            ld = lSubData[:, nonRNullRows]

            lOut = self.layerModel.fprop(ld)
            rOut = self.layerModel.fprop(rd)
            sigOut = sigmoid(rOut - lOut);
            d_outputs = sigOut;
            if len(d_outputs > 0) and d_outputs.shape[1] > 0:
                self.layerModel.bprop(d_outputs, rd)
                currGradContrib = self.layerModel.grad(d_outputs, rd)
                currGradContrib = currGradContrib.sum(1)
                currGrad += currGradContrib;

        #myLog.info('problemArr.shape[0]: %d' % problemArr.shape[0])
        decayContribution = 2 * self.l2decay * self.params * problemArr.shape[0]
        #myLog.info('min, max decayContribution: %0.4f %0.4f' % (min(decayContribution), max(decayContribution)))
        
        currGrad += decayContribution;
        if self.chunklog:
            myLog.debug('||currGrad||^1 : %.4f, ||decayContribution|| : %.4f, mean(currGrad) : %.4f, max(currGrad) : %.4f' \
                        % (abs(currGrad).sum(), self.l2decay * (self.params**2).sum() * problemArr.shape[0],
                           mean(currGrad), max(abs(currGrad))));
        return currGrad;
    
    
    def apply(self, newData):
        """Apply the trained neural net to this new data"""
        outData = zeros(len(newData));
        currIdx = 0;
        for lDataChunk in self.singleSideGradChunkDataIterator(newData):
            actOut = self.layerModel.fprop(lDataChunk);
            actOut = actOut.flatten();
            outData[currIdx:(currIdx + len(actOut))] += actOut;
            currIdx += len(actOut);
        return outData.flatten();
    
if __name__ == '__main__':
    ## Not meant to be called from cmd line directly
    sys.exit(-1);
