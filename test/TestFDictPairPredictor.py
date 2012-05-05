#!/usr/bin/env python
# encoding: utf-8
"""
TestFDictPairPredictor.py

Created by Matt Kayala on 2010-10-18.
"""
import sys, os;
import unittest;
import cStringIO;
import tempfile;
import gzip;
from pprint import pformat;

from nnutils.Util import FeatureDictWriter;
from nnutils.mutil.Const import EPSILON
from nnutils.mutil.MonteArchModel import MonteArchModel, saveArchModel;
from nnutils.rank.FDictPairPredictor import FDictPairPredictor;

from numpy import zeros, ones, concatenate, array, multiply;
from numpy.random import randn
from numpy import max, min;

import Const, Util;
from Util import log;

class TestFDictPairPredictor(unittest.TestCase):
    def setUp(self):
        """Set up anything for the tests.., """
        super(TestFDictPairPredictor, self).setUp();
        
        (self.ARCH_FD, self.ARCH_FILENAME) = tempfile.mkstemp();
        (self.FEAT_FD, self.FEAT_FILENAME) = tempfile.mkstemp();
        (self.OUT_FD, self.OUT_FILENAME) = tempfile.mkstemp();
        
        self.LOW_DIST_MEANS = array([10, -10, 5, -5]);
        #self.LOW_DIST_MEANS = array([100, -100, 50, -50]);
        self.LOW_DIST_VARS = array([5, 5, 5, 5])
        
        self.HIGH_DIST_MEANS = array([3, 2, 0, 5]);
        self.HIGH_DIST_VARS = array([3, 2, 5, 5]);
        
        self.NUM_PAIRS = 1000;
        self.lowFeatDictList = self.__generateFeatDictList(self.NUM_PAIRS, self.LOW_DIST_MEANS, self.LOW_DIST_VARS);
        self.highFeatDictList = self.__generateFeatDictList(self.NUM_PAIRS, self.HIGH_DIST_MEANS, self.HIGH_DIST_VARS);
        
        self.FEAT_DATA = [];
        self.FEAT_DATA.extend(self.lowFeatDictList)
        self.FEAT_DATA.extend(self.highFeatDictList)
        
        # Write out the fdict data
        ofs = gzip.open(self.FEAT_FILENAME, 'w')
        writer = FeatureDictWriter(ofs)
        for iLine, fd in enumerate(self.FEAT_DATA):
            writer.update(fd, str(iLine))
        ofs.close();
        
        self.EXPECTED_NUM_OUTPUTLINES = 2 * self.NUM_PAIRS;
        
        # Set up an arch model:
        archModel = MonteArchModel();
        archModel.paramVar = 0.01
        archModel.numhidden = 0;
        archModel.numfeats = len(self.LOW_DIST_MEANS);
        archModel.l2decay = 0.001;
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
        
        saveArchModel(self.ARCH_MDL, self.ARCH_FILENAME);
    
    
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
    
    
    def tearDown(self):
        """Restore state"""
        os.close(self.ARCH_FD)
        os.close(self.OUT_FD)
        os.close(self.FEAT_FD)
        
        os.remove(self.ARCH_FILENAME)
        os.remove(self.OUT_FILENAME)
        os.remove(self.FEAT_FILENAME)
        
        super(TestFDictPairPredictor, self).tearDown();
    
    
    def test_main(self):
        """Test that the main function works as expected"""
        instance = FDictPairPredictor()
        args = ['', self.ARCH_FILENAME, self.FEAT_FILENAME, self.OUT_FILENAME]
        instance.main(args)
        
        self.assert_(all(abs(instance.archModel.params - self.ARCH_MDL.params) < EPSILON))
        
        ifs = open(self.OUT_FILENAME)
        data = [float(line.strip()) for line in ifs]
        ifs.close()
        
        self.assertEqual(len(data), len(self.FEAT_DATA))
        

def suite():
    """Returns the suite of tests to run for this test class / module.
    Use unittest.makeSuite methods which simply extracts all of the
    methods for the given class whose name starts with "test"

    Actually, since this is mostly all basic input / output functions,
    can do most of it with doctests and DocTestSuite
    """
    suite = unittest.TestSuite();
    suite.addTest(unittest.makeSuite(TestFDictPairPredictor));
    return suite;

if __name__=="__main__":
    unittest.TextTestRunner(verbosity=Const.RUNNER_VERBOSITY).run(suite())
