#!/usr/bin/env python
# encoding: utf-8
"""
Test_MonteFeatDictPredictor.py

Created by Matt Kayala on 2010-10-10.

A Test Case in the CHEM Module.
"""
import sys, os;
import unittest;
import cStringIO;
import tempfile;
import cPickle;
import csv;
import gzip;
import pprint;

from nnutils.Util import FeatureDictWriter;
from nnutils.mutil.MonteArchModel import MonteArchModel, loadArchModel, saveArchModel;
from nnutils.mutil.MonteFeatDictClassifier import MonteFeatDictClassifier;
from nnutils.mutil.MonteFeatDictPredictor import MonteFeatDictPredictor;

import Const, Util;
from Util import log;

from numpy import zeros, ones, concatenate, array, multiply, allclose;
from numpy.random import randn;

from numpy import max, min;

class TestMonteFeatDictPredictor(unittest.TestCase):
    def setUp(self):
        """Set up anything for the tests.., """
        super(TestMonteFeatDictPredictor, self).setUp();
        
        # a file for the archmodel out
        (self.AMODELIN_FD, self.AMODELIN_FILENAME) = tempfile.mkstemp();
        (self.FDICT_FD, self.FDICT_FILENAME) = tempfile.mkstemp();
        (self.IDX_FD, self.IDX_FILENAME) = tempfile.mkstemp();
        # A file for the complete output
        (self.OUT_FD, self.OUT_FILENAME) = tempfile.mkstemp();
        
        # Set up an ArchModel
        self.NUMFEATS = 3;
        self.NUMDATA_POS = 1000
        self.NUMDATA_NEG = self.NUMDATA_POS;
        self.ARCHMODEL = MonteArchModel();
        self.ARCHMODEL.numfeats = self.NUMFEATS;
        self.ARCHMODEL.numEpochs = 15;
        self.ARCHMODEL.batch = False;
        self.ARCHMODEL.gradientChunkSize=1000;
        self.ARCHMODEL.l2decay = 0;
        self.ARCHMODEL.onlineChunkSize = 5000;
        self.ARCHMODEL.numhidden = 10;
        self.ARCHMODEL.trainertype = 'gdescadapt';
        self.ARCHMODEL.qLearningRate = 0.05;
        self.ARCHMODEL.exponentAvgM = 0.95;
        self.ARCHMODEL.learningrate = 0.1;
        self.ARCHMODEL.setupParams();
        saveArchModel(self.ARCHMODEL, self.AMODELIN_FILENAME);
        
        # Set up a data and targ arr (should be imbalanced.)
        self.POS_MEANS = [-20, -50, -10]
        self.NEG_MEANS = [1, -2, -3]
        self.POS_DATA = self.POS_MEANS * randn(self.NUMDATA_POS, self.NUMFEATS);
        self.NEG_DATA = self.NEG_MEANS * randn(self.NUMDATA_NEG, self.NUMFEATS);
        self.DATA_ARR = concatenate((self.POS_DATA, self.NEG_DATA), 0);
        # (And because all the data is in similar forms)
        self.DATA_ARR /= 10.0;
        
        self.FEAT_DATA = [];
        for iRow in range(self.DATA_ARR.shape[0]):
            d = {};
            for iCol in range(self.DATA_ARR.shape[1]):
                d[iCol] = self.DATA_ARR[iRow,iCol];
            self.FEAT_DATA.append(d);
        
        # Write out the fdict stuff
        ofs = gzip.open(self.FDICT_FILENAME, 'w');
        writer = FeatureDictWriter(ofs);
        for iRow, d in enumerate(self.FEAT_DATA):
            writer.update(d, str(iRow));
        ofs.close();
        
        # construct the idxArr data
        posIdx = range(len(self.POS_DATA));
        negIdx = range(len(self.NEG_DATA), len(self.POS_DATA) + len(self.NEG_DATA));
        
        self.IDX_ARR = [];
        for aPos in posIdx:
            self.IDX_ARR.append([aPos, aPos, 1.0, 0.0]);
        for aNeg in negIdx:
            self.IDX_ARR.append([aNeg, aNeg, 0.0, 1.0]);
        
        ofs = open(self.IDX_FILENAME, 'w');
        writer = csv.writer(ofs, quoting=csv.QUOTE_NONE);
        for line in self.IDX_ARR:
            writer.writerow(line);
        ofs.close();
    
    def tearDown(self):
        """Restore state"""
        # Tear down all the temp files.
        os.close(self.AMODELIN_FD);
        os.close(self.FDICT_FD);
        os.close(self.IDX_FD);
        os.close(self.OUT_FD);
        
        os.remove(self.AMODELIN_FILENAME)
        os.remove(self.FDICT_FILENAME)
        os.remove(self.IDX_FILENAME)
        os.remove(self.OUT_FILENAME)
        
        super(TestMonteFeatDictPredictor, self).tearDown();
    
    
    def test_basic(self):
        """Test of the running.  Test that id's and targets are correct."""
        predictor = MonteFeatDictPredictor();
        predictor.archModel = self.ARCHMODEL;
        predictor.setup();
        
        currRow = 0;
        for retTarg, retPred in predictor.predict(self.FEAT_DATA, self.IDX_ARR):
            self.assertEqual(len(retTarg), len(retPred));
            start = currRow; 
            end = currRow + len(retPred);
            expectedTarg = self.IDX_ARR[start:end];
            for r, e in zip(retTarg, expectedTarg):
                r = array(r)
                e = array(e)
                self.assert_(all(e == r));
            self.assert_(all(retPred <= 1))
            self.assert_(all(retPred >= 0));
            currRow = end;
        
        self.assertEqual(currRow, len(self.IDX_ARR));
    
    
    def test_main(self):
        """Test that the main function works as expected."""
        predictor = MonteFeatDictPredictor();
        args = ['', '--delim=,',  
                self.AMODELIN_FILENAME, self.FDICT_FILENAME, self.IDX_FILENAME, self.OUT_FILENAME];
        predictor.main(args);
        
        ifs = open(self.OUT_FILENAME);
        reader = csv.reader(ifs, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC);
        
        iRow = -1;
        for iRow, row in enumerate(reader):
            expectedTarg = array(self.IDX_ARR[iRow]);
            actualTarg = array(row[:-1]);
            self.assert_(allclose(expectedTarg, actualTarg));
            self.assert_(row[-1] <= 1)
            self.assert_(row[-1] >= 0)
        ifs.close();
        
        self.assertEqual(iRow +1, len(self.IDX_ARR));


def suite():
    """Returns the suite of tests to run for this test class / module.
    Use unittest.makeSuite methods which simply extracts all of the
    methods for the given class whose name starts with "test"

    Actually, since this is mostly all basic input / output functions,
    can do most of it with doctests and DocTestSuite
    """
    suite = unittest.TestSuite();
    suite.addTest(unittest.makeSuite(TestMonteFeatDictPredictor));
    return suite;

if __name__=="__main__":
    unittest.TextTestRunner(verbosity=Const.RUNNER_VERBOSITY).run(suite())
