#!/usr/bin/env python
# encoding: utf-8
"""
TestMonteArchModel.py

Created by Matt Kayala on 2010-05-06.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.

A Test Case in the CHEM Module.
"""
import sys, os;
import unittest;
import cStringIO;
import tempfile;


from nnutils.mutil import MonteArchModel;

import Const, Util;
from Util import log;

from nnutils.monte.bp.neuralnet import NNlayer, SigmoidLinear;

class TestMonteArchModel(unittest.TestCase):
    def setUp(self):
        """Set up anything for the tests.., """
        super(TestMonteArchModel, self).setUp();
        
        (self.FILE_FD, self.FILE_FILENAME) = tempfile.mkstemp();
    
    
    def tearDown(self):
        """Restore state"""
        os.close(self.FILE_FD)
        os.remove(self.FILE_FILENAME);
        
        super(TestMonteArchModel, self).tearDown();
    
    
    def test_MonteArchModelCreateLoad(self):
        """Test that the code to create and then reload a monte arch object works."""
        numFeats = 100;
        numHidden = 10;
        gradientChunkSize = 100;
        
        expNumParams = SigmoidLinear.numparams(numFeats, numHidden, 1);
        
        args = ['', '-n', str(numHidden), '-f', str(numFeats), 
            '--gradientChunkSize', str(gradientChunkSize), self.FILE_FILENAME]
        
        instance = MonteArchModel.MonteArchModelMaker()
        instance.main(args);
        
        model = MonteArchModel.loadArchModel(self.FILE_FILENAME);
        
        self.assertEquals(model.numfeats, numFeats)
        self.assertEquals(model.numhidden, numHidden);
        self.assertEquals(model.gradientChunkSize, gradientChunkSize);
        self.assertEquals(model.numparams, expNumParams);
        self.assertEquals(model.params.shape, (expNumParams,));
        
        


def suite():
    """Returns the suite of tests to run for this test class / module.
    Use unittest.makeSuite methods which simply extracts all of the
    methods for the given class whose name starts with "test"

    Actually, since this is mostly all basic input / output functions,
    can do most of it with doctests and DocTestSuite
    """
    suite = unittest.TestSuite();
    suite.addTest(unittest.makeSuite(TestMonteArchModel));
    return suite;

if __name__=="__main__":
    unittest.TextTestRunner(verbosity=Const.RUNNER_VERBOSITY).run(suite())
