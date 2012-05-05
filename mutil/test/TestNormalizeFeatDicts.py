#!/usr/bin/env python
# encoding: utf-8
"""
TestNormalizeFeatDicts.py

Created by Matt Kayala on 2010-10-27.
"""
import sys, os;
import unittest;
import cStringIO;
import tempfile;
import gzip;
from pprint import pformat;
import string;
import cPickle

from nnutils.Util import FeatureDictReader, FeatureDictWriter;
from nnutils.mutil.NormalizeFeatDicts import NormalizeFeatDicts;

import Const, Util;
from Util import log, spectrumExtractor

class TestNormalizeFeatDicts(unittest.TestCase):
    def setUp(self):
        """Set up anything for the tests.., """
        super(TestNormalizeFeatDicts, self).setUp();
        
        (self.MAP_FD, self.MAP_FILENAME) = tempfile.mkstemp();
        (self.FEAT_FD, self.FEAT_FILENAME) = tempfile.mkstemp(suffix='.gz');
        (self.TE_FEAT_FD, self.TE_FEAT_FILENAME) = tempfile.mkstemp(suffix='.gz')
        (self.NORM_PAR_FD, self.NORM_PAR_FILENAME) = tempfile.mkstemp();
        
        (self.OUT_TR_F_FD, self.OUT_TR_F_FILENAME) = tempfile.mkstemp(suffix='.gz');
        (self.OUT_TE_F_FD, self.OUT_TE_F_FILENAME) = tempfile.mkstemp(suffix='.gz');
        
        # Data items to build feature dictionaries for.  Just build several random alphanumeric strings
        self.trdataList = \
            [   ("L648oAqdRYZQMUx2RO1R", 1, 0),
                ("XyGEXXbrhKxWXGC7UlIG", 2, 0),
                ("TJMS6V6GtYMXZCVzeyWk", 3, 0),
                ("wrIx1VarDGoPL5YwM2tX", 4, 0),
                ("GfvlJThPPjaRXBYz7F6B", 5, 0),
                ("khsdlahsdhsdjhah7s77", 6, 0),
                ("mqScDuJQBjRASbvE5AXQ", 7, 1),
                ("FSAmpXTabhpi6iJNhb1D", 8, 1)
            ];
        self.tedatalist = \
            [
                ("PcIs3uMZDA3UTraaBhwh", 10, 1),
                ("GNSgYBWZ44yzxjDaMkyE", 11, 0),
                ("3zVXf7Gb03wgORz9KmPs", 12, 1)
            ]
        
        #
        # Then build a featureMapFile
        self.featureList = string.ascii_letters;
        self.featureList += string.digits;
        self.mapObj = {};
        for iFeat, feature in enumerate(self.featureList):
            self.mapObj[feature] = iFeat;
        self.mapObj['testOther'] = len(self.featureList)
        
        ofs = open(self.MAP_FILENAME, 'w')
        cPickle.dump(self.mapObj, ofs);
        ofs.close();
        
        # Then build the featDict files and the class file
        # Simple kernel to build the feature dictionaries to test
        # (This simply returns counts over characters)
        self.kernel = spectrumExtractor
        
        trFeatOFS = gzip.open(self.FEAT_FILENAME, 'w')
        trFeatWriter = FeatureDictWriter(trFeatOFS);
        self.trainIdList = [];
        for iLine, (theString, idx, classVal) in enumerate(self.trdataList):
            self.trainIdList.append((iLine, idx, classVal));
            featureDict = self.kernel(theString)
            if classVal > 0:
                featureDict['testOther'] = classVal;
            trFeatWriter.update(featureDict, str(idx))
        
        trFeatOFS.close();
        
        teFeatOFS = gzip.open(self.TE_FEAT_FILENAME, 'w');
        teFeatWriter = FeatureDictWriter(teFeatOFS);
        for iLine, (theString, idx, classVal) in enumerate(self.tedatalist):
            featureDict = self.kernel(theString);
            if classVal > 0:
                featureDict['testOther'] = classVal;
            teFeatWriter.update(featureDict, str(idx));
        teFeatOFS.close();
        
        # The 'testOther' feature should have a minRange of (0, 1)
        self.EXP_TEST_OTHER_MINRANGE = [0,1.0];
    
    
    def tearDown(self):
        """Restore state"""
        os.close(self.TE_FEAT_FD);
        os.close(self.FEAT_FD);
        os.close(self.MAP_FD);
        os.close(self.OUT_TR_F_FD)
        os.close(self.OUT_TE_F_FD)
        os.close(self.NORM_PAR_FD)
        
        os.remove(self.TE_FEAT_FILENAME)
        os.remove(self.FEAT_FILENAME)
        os.remove(self.MAP_FILENAME)
        os.remove(self.OUT_TR_F_FILENAME)
        os.remove(self.OUT_TE_F_FILENAME)
        os.remove(self.NORM_PAR_FILENAME);
        
        super(TestNormalizeFeatDicts, self).tearDown();

    
    def test_processFeatDicts(self):
        """Test that the basic processing of feature dicts works as expected"""
        ifs = gzip.open(self.FEAT_FILENAME)
        reader = FeatureDictReader(ifs)
        instance = NormalizeFeatDicts();
        instance.loadFeatKeyToColMap(self.MAP_FILENAME);
        
        sMinMaxDict = instance.processFeatureDictList(reader);
        self.assertEqual(sMinMaxDict['testOther'], self.EXP_TEST_OTHER_MINRANGE);
        
        self.assert_(len(sMinMaxDict.keys()) <= len(self.featureList))
        #log.info('The MinMaxDict is %s' % pformat(sMinMaxDict));
        ifs.close()
        
        ifs = gzip.open(self.FEAT_FILENAME)
        reader = FeatureDictReader(ifs)
        newFDictList = [];
        theIdList = [];
        for idx, fdict in instance.normalizeFeatDictList(reader, sMinMaxDict, self.mapObj):
            newFDictList.append(fdict)
            theIdList.append(idx)
        ifs.close();
        
        self.assertEqual(len(newFDictList), len(self.trdataList))
        self.assertEqual(len(theIdList), len(self.trdataList));
        
        #log.info('Example normFDict: %s' % pformat(newFDictList[0]));
    
    def test_main(self):
        """Test that calling the main function works as expected"""
        instance = NormalizeFeatDicts();
        
        args = ['', '-p', self.NORM_PAR_FILENAME, '-t', self.TE_FEAT_FILENAME, '-o', self.OUT_TE_F_FILENAME, 
                    self.MAP_FILENAME, self.FEAT_FILENAME, self.OUT_TR_F_FILENAME];
        
        instance.main(args);
        
        # Check on the featdict files
        # First the train data
        ifs = gzip.open(self.OUT_TR_F_FILENAME)
        reader = FeatureDictReader(ifs);
        fDictList = [f for f in reader]
        ifs.close();
        self.assertEqual(len(self.trdataList), len(fDictList))
        #log.info('A sample train featdict : %s' % pformat(fDictList[0]))
        
        
        ifs = gzip.open(self.OUT_TE_F_FILENAME)
        reader = FeatureDictReader(ifs);
        fDictList = [f for f in reader]
        ifs.close();
        self.assertEqual(len(self.tedatalist), len(fDictList))
        #log.info('A sample test featdict : %s' % pformat(fDictList[0]));
        
    
    

def suite():
    """Returns the suite of tests to run for this test class / module.
    Use unittest.makeSuite methods which simply extracts all of the
    methods for the given class whose name starts with "test"

    Actually, since this is mostly all basic input / output functions,
    can do most of it with doctests and DocTestSuite
    """
    suite = unittest.TestSuite();
    suite.addTest(unittest.makeSuite(TestNormalizeFeatDicts));
    return suite;

if __name__=="__main__":
    unittest.TextTestRunner(verbosity=Const.RUNNER_VERBOSITY).run(suite())
