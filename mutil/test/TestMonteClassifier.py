#!/usr/bin/env python
# encoding: utf-8
"""
TestMonteClassifier.py

Created by Matt Kayala on 2010-05-09.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.

A Test Case in the CHEM Module.
"""
import sys, os;
import unittest;
import cStringIO;
import tempfile;
import pprint;

from nnutils.mutil.MonteClassifier import MonteClassifier;
from nnutils.mutil.MonteArchModel import MonteArchModel;
from nnutils.mutil.Util import rmse, accuracy;

from numpy import zeros, ones, concatenate, array, multiply, log, memmap;
from numpy.random import randn

from numpy import max, min;



import Const, Util;
from Util import log as myLogger;

class TestMonteClassifier(unittest.TestCase):
    def setUp(self):
        """Set up anything for the tests.., """
        super(TestMonteClassifier, self).setUp();
        
        # Setup some fake data
        numFeats = 3;
        sizePosData = 2000;
        sizeNegData = 2000;
        
        posMeans = array([-20, -50, -10 ])
        negMeans = array([1, -2, -3 ])
        
        fakePosData = posMeans + randn(sizePosData, numFeats);
        fakeNegData = negMeans + randn(sizeNegData, numFeats);
        
        fakePosTarg = ones(sizePosData)
        fakeNegTarg = zeros(sizeNegData);
        
        self.DATA_ARR = concatenate((fakePosData, fakeNegData), 0);
        maxByCol = max(self.DATA_ARR, 0)
        minByCol = min(self.DATA_ARR, 0)
        rangeByCol = maxByCol - minByCol + 1E-8;
        self.DATA_ARR -= minByCol;
        self.DATA_ARR /= rangeByCol;
        self.TARG_ARR = concatenate((fakePosTarg, fakeNegTarg), 0);
        
        # Set up an arch model:
        archModel = MonteArchModel();
        archModel.paramVar = 0.01
        archModel.numhidden = 0;
        archModel.numfeats = numFeats;
        archModel.l2decay = 0.01;
        archModel.gradientChunkSize = 50;
        archModel.onlineChunkSize = 4000;
        archModel.cgIterations = 2;
        archModel.batch = False;
        archModel.numEpochs = 20;
        archModel.setupParams();
        
        self.ARCH_MDL = archModel;
        
    
    
    def tearDown(self):
        """Remove the data, cleanup"""        
        super(TestMonteClassifier, self).tearDown();
    
    def __generalTest(self, dataArr, targArr, archModel):
        """Convenience to test things multiple times"""
        classifier = MonteClassifier(archModel, dataArr, targArr);
        classifier.setupModels();
        
        currOut = classifier.apply(dataArr);
        currCost =  -(multiply(targArr, log(currOut)) + multiply(1 - targArr, log(1-currOut))).sum();
        decayContrib = classifier.l2decay * (classifier.params**2).sum();
        currCost += decayContrib;
        currRMSE = rmse(currOut, targArr);
        currAcc = accuracy(currOut, targArr);
        
        myLogger.info('(Cost, Acc, RMSE, decay) before training: (%.4f, %.4f, %.4f, %.4f)' % (currCost, currAcc, currRMSE, decayContrib));
        #myLogger.info('head(targArr) : %s, tail(targArr) : %s' % (pprint.pformat(targArr[:5]), pprint.pformat(targArr[-5:])))
        #myLogger.info('head(dataArr) : %s, tail(dataArr) : %s' % (pprint.pformat(dataArr[:5, :]), pprint.pformat(dataArr[-5:, :])))        
        #myLogger.info('head(currOut) : %s, tail(currOut) : %s' % (pprint.pformat(currOut[:5]), pprint.pformat(currOut[-5:])))
        myLogger.info('Starting params : %s' % pprint.pformat(classifier.params))
        
        classifier.train();
        
        fOut = classifier.apply(dataArr);
        fCost =  -(multiply(targArr, log(fOut)) + multiply(1 - targArr, log(1-fOut))).sum();
        decayContrib = classifier.l2decay * (classifier.params**2).sum();
        fCost += decayContrib;
        fRMSE = rmse(fOut, targArr);
        fAcc = accuracy(fOut, targArr);
        
        myLogger.info('(Cost, Acc, RMSE, decay) after training: (%.4f, %.4f, %.4f, %.4f)' % (fCost, fAcc, fRMSE, decayContrib));
        myLogger.info('Final params : %s' % pprint.pformat(classifier.params))
        
        self.assert_(fRMSE < currRMSE, 'RMSE did not decrease.')
    
    
    def test_basic(self):
        """Basic test that the classifier code works as expected with a simple array and simple classification 
        problem"""
        self.__generalTest(self.DATA_ARR, self.TARG_ARR, self.ARCH_MDL);
    
    
    def _test_basicBatch(self):
        """Test that the classifier code works as expected with a batch learner.  (all mem array)"""
        self.ARCH_MDL.batch = True;
        self.ARCH_MDL.trainertype = 'bfgs';
        self.__generalTest(self.DATA_ARR, self.TARG_ARR, self.ARCH_MDL);


def suite():
    """Returns the suite of tests to run for this test class / module.
    Use unittest.makeSuite methods which simply extracts all of the
    methods for the given class whose name starts with "test"

    Actually, since this is mostly all basic input / output functions,
    can do most of it with doctests and DocTestSuite
    """
    suite = unittest.TestSuite();
    suite.addTest(unittest.makeSuite(TestMonteClassifier));
    return suite;

if __name__=="__main__":
    unittest.TextTestRunner(verbosity=Const.RUNNER_VERBOSITY).run(suite())
