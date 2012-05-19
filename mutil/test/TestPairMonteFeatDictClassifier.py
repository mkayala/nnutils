#!/usr/bin/env python
# encoding: utf-8
"""
TestPairMonteFeatDictClassifier.py

Test that the code to run the pairwise feat dict classification works as expected.

Created by Matt Kayala on 2010-08-12.
"""
import sys, os;
import unittest;
import cStringIO;
import tempfile;

import pprint;

from nnutils.mutil.PairMonteFeatDictClassifier import PairMonteFeatDictClassifier;
from nnutils.mutil.MonteArchModel import MonteArchModel;
from nnutils.mutil.Util import rmse, accuracy, sigmoid;
from nnutils.mutil.Const import OFFSET_EPSILON;

from numpy import zeros, ones, concatenate, array, multiply, log, memmap, where;
from numpy.random import randn
from numpy import max, min;

import Const, Util;
from Util import log as myLogger;

class TestPairMonteFeatDictClassifier(unittest.TestCase):
    def setUp(self):
        """Set up anything for the tests.., """
        super(TestPairMonteFeatDictClassifier, self).setUp();
        
        self.LOW_DIST_MEANS = array([10, -10, 5, -5]);
        self.LOW_DIST_VARS = array([5, 5, 5, 5])
        
        self.HIGH_DIST_MEANS = array([3, 2, 0, 5]);
        self.HIGH_DIST_VARS = array([3, 2, 5, 5]);
        
        self.NUM_PAIRS = 10000
        self.lowFeatDictList = self.__generateFeatDictList(self.NUM_PAIRS, self.LOW_DIST_MEANS, self.LOW_DIST_VARS);
        self.highFeatDictList = self.__generateFeatDictList(self.NUM_PAIRS, self.HIGH_DIST_MEANS, self.HIGH_DIST_VARS);
        
        self.FEAT_DATA, self.PROB_ARR = self.__generateProbMatFeatDictList(self.highFeatDictList, self.lowFeatDictList);
        self.FEAT_DATA_N, self.PROB_ARR_N = self.__generateProbMatFeatDictList(self.highFeatDictList, self.lowFeatDictList, nullVal=True);
                
        numFeats = len(self.HIGH_DIST_VARS);
        
        # Set up an arch model:
        archModel = MonteArchModel();
        archModel.paramVar = 0.01
        archModel.numhidden = 0;
        archModel.numfeats = numFeats;
        archModel.l2decay = 1.0;
        archModel.gradientChunkSize = 500;
        archModel.onlineChunkSize = 4000;
        archModel.cgIterations = 2;
        archModel.batch = False;
        archModel.numEpochs = 50;
        archModel.trainertype = 'gdescadapt'
        archModel.qLearningRate = 0.05;
        archModel.exponentAvgM = 0.95;
        archModel.learningrate = 0.1;
        archModel.setupParams();
        self.ARCH_MDL = archModel;
    
    
    def tearDown(self):
        """Restore state"""
        super(TestPairMonteFeatDictClassifier, self).tearDown();
        
    def __generateFeatDictList(self, numPoints, featMeans, featVars):
        """Convenience to generate a list of featDicts, where each of the features are norm distributed with the 
        parameters"""
        featData = randn(numPoints, len(featMeans));
        featData = featData * featVars + featMeans;
        
        featDictList = []
        for arr in featData:
            theDict = {};
            for i in range(len(arr)):
                theDict[i] = arr[i];
            featDictList.append(theDict);
        
        return featDictList;
    
    def __generateProbMatFeatDictList(self, featDictList1, featDictList2, nullVal=False):
        """Convenience to take two feat dict lists and return the featDictList, probMat to be able to write out 
        as input. """
        if nullVal:
            probMat = zeros((len(featDictList1), 2), dtype=int)
            probMat[:, 0] = range(len(featDictList1));
            probMat[:, 1] = probMat[:, 0] + len(featDictList1)

            pm2 = zeros((len(featDictList1), 2), dtype=int)
            pm2[:, 0] = range(len(featDictList1));
            pm2[:, 1] = -1

            pm3 = zeros((len(featDictList1), 2), dtype=int)
            pm3[:, 1] = probMat[:, 1]
            pm3[:, 0] = -1
            #probMat = concatenate(( probMat, pm2, pm3))
            probMat = concatenate(( probMat, probMat, pm2, pm3))
            #probMat = concatenate((probMat, probMat, probMat, pm2, pm3))
        else:
            probMat = zeros((len(featDictList1), 2), dtype=int)
            probMat[:, 0] = range(len(featDictList1));
            probMat[:, 1] = probMat[:, 0] + len(featDictList1)
        
        retFeatDictList = [];
        retFeatDictList.extend(featDictList1)
        retFeatDictList.extend(featDictList2);
        
        return (retFeatDictList, probMat);
    
    def __generalTest(self, featDictList, probArr, archModel):
        """Convenience to test things multiple times"""
        classifier = PairMonteFeatDictClassifier(archModel, featDictList, probArr);
        classifier.setupModels();
        
        currOut = classifier.apply(featDictList);
        
        lData = currOut[probArr[:, 0]]
        rData = currOut[probArr[:, 1]]
        
        outputs = sigmoid(lData - rData);
        outputs = where(outputs < OFFSET_EPSILON, OFFSET_EPSILON, outputs);
        outputs = where(outputs > 1-OFFSET_EPSILON, 1-OFFSET_EPSILON, outputs);
        
        # Assuming all is 1
        currCost =  2 * -(log(outputs)).sum();
        decayContrib = classifier.l2decay * (classifier.params**2).sum();
        currCost += decayContrib;
        currRMSE = rmse(outputs, 1);
        currAcc = accuracy(outputs, 1);
        
        beginRMSE = currRMSE;
        
        myLogger.info('(Cost, Acc, RMSE, decay) before training: (%.4f, %.4f, %.4f, %.4f)' % (currCost, currAcc, currRMSE, decayContrib));
        
        classifier.train();
        
        currOut = classifier.apply(featDictList);
        
        lData = currOut[probArr[:, 0]]
        rData = currOut[probArr[:, 1]]
        
        outputs = sigmoid(lData - rData);
        outputs = where(outputs < OFFSET_EPSILON, OFFSET_EPSILON, outputs);
        outputs = where(outputs > 1-OFFSET_EPSILON, 1-OFFSET_EPSILON, outputs);
        
        # Assuming all is 1
        currCost =  -(log(outputs)).sum();
        decayContrib = classifier.l2decay * (classifier.params**2).sum();
        currCost += decayContrib;
        currRMSE = rmse(outputs, 1);
        currAcc = accuracy(outputs, 1);
        
        myLogger.info('(Cost, Acc, RMSE, decay) after training: (%.4f, %.4f, %.4f, %.4f)' % (currCost, currAcc, currRMSE, decayContrib));
        
        self.assert_(currRMSE < beginRMSE, 'RMSE did not decrease.');
    
    
    
    def _test_GDescApapt(self):
        """Test of simple run through using gdescadapt."""
        self.ARCH_MDL.batch = False;    
        self.ARCH_MDL.trainertype = 'gdescadapt';
        #self.ARCH_MDL.trainertype = 'gdesc';
        self.ARCH_MDL.onlineChunkSize = 500;
        self.ARCH_MDL.qLearningRate = 0.01;
        self.ARCH_MDL.exponentAvgM = 0.95;
        self.ARCH_MDL.numhidden = 10;
        self.ARCH_MDL.paramVar = 0.0001;
        self.ARCH_MDL.setupParams();
        
        self.ARCH_MDL.gradientChunkSize = 100;
        self.ARCH_MDL.l2decay = 10.0;
        self.ARCH_MDL.numEpochs = 5;
        self.ARCH_MDL.learningrate = 0.01;
        self.__generalTest(self.FEAT_DATA, self.PROB_ARR, self.ARCH_MDL);

    def test_GDescApapt_NULLVal(self):
        """Test of training with a NULL value (-1 id in probMat) to represent a 0-vector input"""
        self.ARCH_MDL.batch = False;    
        self.ARCH_MDL.trainertype = 'gdescadapt';
        #self.ARCH_MDL.trainertype = 'gdesc';
        self.ARCH_MDL.onlineChunkSize = 2000;
        self.ARCH_MDL.qLearningRate = 0.01;
        self.ARCH_MDL.exponentAvgM = 0.95;
        self.ARCH_MDL.numhidden = 3;
        self.ARCH_MDL.paramVar = 0.0001;
        self.ARCH_MDL.setupParams();
        
        self.ARCH_MDL.gradientChunkSize = 100;
        self.ARCH_MDL.l2decay = 1000.0
        self.ARCH_MDL.numEpochs = 10;
        self.ARCH_MDL.learningrate = 0.01;
        self.__generalTest(self.FEAT_DATA_N, self.PROB_ARR_N, self.ARCH_MDL);

    
    
def suite():
    """Returns the suite of tests to run for this test class / module.
    Use unittest.makeSuite methods which simply extracts all of the
    methods for the given class whose name starts with "test"

    Actually, since this is mostly all basic input / output functions,
    can do most of it with doctests and DocTestSuite
    """
    suite = unittest.TestSuite();
    suite.addTest(unittest.makeSuite(TestPairMonteFeatDictClassifier));
    return suite;

if __name__=="__main__":
    unittest.TextTestRunner(verbosity=Const.RUNNER_VERBOSITY).run(suite())
