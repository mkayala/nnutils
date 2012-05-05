#!/usr/bin/env python
# encoding: utf-8
"""
TestMonteFeatDictClassifier.py

Created by Matt Kayala on 2010-06-16.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.

A Test Case in the CHEM Module.
"""
import sys, os;
import unittest;
import cStringIO;
import tempfile;
import pprint;

from nnutils.mutil.MonteFeatDictClassifier import MonteFeatDictClassifier;
from nnutils.mutil.MonteArchModel import MonteArchModel;
from nnutils.mutil.Util import rmse, accuracy;

from numpy import zeros, ones, concatenate, array, multiply, log, memmap;
from numpy.random import randn

from numpy import max, min;

import Const, Util;
from Util import log as myLogger;

class TestMonteFeatDictClassifier(unittest.TestCase):
    def setUp(self):
        """Set up anything for the tests.., """
        super(TestMonteFeatDictClassifier, self).setUp();
        
        # Setup some fake data
        numFeats = 3;
        sizePosData = 10000;
        sizeNegData = 10000;
        
        posMeans = array([-2, 5, 6 ])
        negMeans = array([.1, -.2, -.3 ])
        
        fakePosData = posMeans + randn(sizePosData, numFeats)*.1;
        fakeNegData = negMeans + randn(sizeNegData, numFeats)*.1;
        
        fakePosTarg = ones(sizePosData)
        fakeNegTarg = zeros(sizeNegData);
        
        self.DATA_ARR = concatenate((fakePosData, fakeNegData), 0);
        
        self.FEAT_DATA = [];
        for iRow in range(self.DATA_ARR.shape[0]):
            d = {};
            for iCol in range(self.DATA_ARR.shape[1]):
                d[iCol] = self.DATA_ARR[iRow,iCol];
            self.FEAT_DATA.append(d);
        
        self.TARG_ARR = concatenate((fakePosTarg, fakeNegTarg), 0);
        
        self.IDX_ARR = array(range(len(self.TARG_ARR)));
        
        # Set up an arch model:
        archModel = MonteArchModel();
        archModel.paramVar = 0.01
        archModel.numhidden = 0;
        archModel.numfeats = numFeats;
        archModel.l2decay = 0.001;
        archModel.gradientChunkSize = 500;
        archModel.onlineChunkSize = 1000;
        archModel.cgIterations = 2;
        archModel.batch = False;
        archModel.numEpochs = 50;
        archModel.trainertype = 'gdescadapt'
        archModel.qLearningRate = 0.05;
        archModel.exponentAvgM = 0.95;
        archModel.learningrate = 1;
        archModel.setupParams();
        self.ARCH_MDL = archModel;
    
    
    def tearDown(self):
        """Remove the data, cleanup"""
        super(TestMonteFeatDictClassifier, self).tearDown();
    
    
    def __generalTest(self, featDictList, targArr, archModel, idxArr):
        """Convenience to test things multiple times"""
        classifier = MonteFeatDictClassifier(archModel, featDictList, targArr, idxArr);
        classifier.setupModels();
        
        currOut = classifier.apply(featDictList);
        currCost =  -(multiply(targArr, log(currOut)) + multiply(1 - targArr, log(1-currOut))).sum();
        decayContrib = classifier.l2decay * (classifier.params**2).sum();
        currCost += decayContrib;
        currRMSE = rmse(currOut, targArr);
        currAcc = accuracy(currOut, targArr);
        
        myLogger.info('(Cost, Acc, RMSE, decay) before training: (%.4f, %.4f, %.4f, %.4f)' % (currCost, currAcc, currRMSE, decayContrib));
        #myLogger.info('head(targArr) : %s, tail(targArr) : %s' % (pprint.pformat(targArr[:5]), pprint.pformat(targArr[-5:])))
        #myLogger.info('head(dataArr) : %s, tail(dataArr) : %s' % (pprint.pformat(featDictList[:5]), pprint.pformat(featDictList[-5:])))        
        #myLogger.info('head(currOut) : %s, tail(currOut) : %s' % (pprint.pformat(currOut[:5]), pprint.pformat(currOut[-5:])))
        #myLogger.info('Starting params : %s' % pprint.pformat(classifier.params))
        
        classifier.train();
        
        fOut = classifier.apply(featDictList);
        fCost =  -(multiply(targArr, log(fOut)) + multiply(1 - targArr, log(1-fOut))).sum();
        decayContrib = classifier.l2decay * (classifier.params**2).sum();
        fCost += decayContrib;
        fRMSE = rmse(fOut, targArr);
        fAcc = accuracy(fOut, targArr);
        
        myLogger.info('(Cost, Acc, RMSE, decay) after training: (%.4f, %.4f, %.4f, %.4f)' % (fCost, fAcc, fRMSE, decayContrib));
        #myLogger.info('head(fOut) : %s, tail(fOut) : %s' % (pprint.pformat(fOut[:5]), pprint.pformat(fOut[-5:])))
        myLogger.info('Final params : %s' % pprint.pformat(classifier.params))
        
        self.assert_(fRMSE < currRMSE, 'RMSE did not decrease.')
    
    def test_GDescApapt(self):
        """Test of simple run through using ."""
        self.ARCH_MDL.batch = False;    
        self.ARCH_MDL.trainertype = 'gdescadapt';
        self.ARCH_MDL.onlineChunkSize = 1000;
        self.ARCH_MDL.qLearningRate = 0.05;
        self.ARCH_MDL.exponentAvgM = 0.95;
        self.ARCH_MDL.numhidden = 10;
        self.ARCH_MDL.paramVar = 0.001;
        self.ARCH_MDL.setupParams();
        
        self.ARCH_MDL.gradientChunkSize = 500;
        self.ARCH_MDL.l2decay = 0.01;
        self.ARCH_MDL.numEpochs = 10;
        self.ARCH_MDL.learningrate = 1;
        self.__generalTest(self.FEAT_DATA, self.TARG_ARR, self.ARCH_MDL, self.IDX_ARR);


def suite():
    """Returns the suite of tests to run for this test class / module.
    Use unittest.makeSuite methods which simply extracts all of the
    methods for the given class whose name starts with "test"

    Actually, since this is mostly all basic input / output functions,
    can do most of it with doctests and DocTestSuite
    """
    suite = unittest.TestSuite();
    suite.addTest(unittest.makeSuite(TestMonteFeatDictClassifier));
    return suite;

if __name__=="__main__":
    unittest.TextTestRunner(verbosity=Const.RUNNER_VERBOSITY).run(suite())
