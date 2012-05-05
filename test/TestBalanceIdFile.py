#!/usr/bin/env python
# encoding: utf-8
"""
TestBalanceIdFile.py

Created by Matt Kayala on 2010-09-17.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.

A Test Case in the CHEM Module.
"""
import sys, os;
import unittest;
import cStringIO;
import tempfile;

from nnutils.BalanceIdFile import BalanceIdFile;

import Const, Util;
from Util import log;

class TestBalanceIdFile(unittest.TestCase):
    def setUp(self):
        """Set up anything for the tests.., """
        super(TestBalanceIdFile, self).setUp();
        
        # Some tmp files
        (self.IN_FD, self.IN_FILENAME) = tempfile.mkstemp();
        (self.OUT_FD, self.OUT_FILENAME) = tempfile.mkstemp();
        
        self.TR_ID_LIST = \
            [
                [0, 1, 0],
                [1, 2, 0],
                [2, 3, 0],
                [3, 4, 0],
                [4, 5, 0],
                [5, 6, 0],
                [6, 7, 1],
                [7, 8, 1]
            ]
        
        self.EXP_BAL_ID_LIST = \
            [
                [0, 1, 0],
                [1, 2, 0],
                [2, 3, 0],
                [3, 4, 0],
                [4, 5, 0],
                [5, 6, 0],
                [6, 7, 1],
                [6, 7, 1],
                [6, 7, 1],
                [6, 7, 1],
                [6, 7, 1],
                [6, 7, 1],
                [7, 8, 1],
                [7, 8, 1],
                [7, 8, 1],
                [7, 8, 1],
                [7, 8, 1],
                [7, 8, 1],
                
            ]
        
        ofs = open(self.IN_FILENAME, 'w')
        for row in self.TR_ID_LIST:
            print >>ofs, ' '.join([str(x) for x in row]);
        ofs.close();
    
    
    def tearDown(self):
        """Restore state"""
        os.close(self.IN_FD)
        os.close(self.OUT_FD)
        
        os.remove(self.IN_FILENAME);
        os.remove(self.OUT_FILENAME);
        
        super(TestBalanceIdFile, self).tearDown();
    
    
    def test_balanceIdDict(self):
        """Test that the code to balance the id list works as expected"""
        instance = BalanceIdFile();
        
        balIdList = [l for l in instance.balanceIdList(self.TR_ID_LIST)];
        self.assertEqual(len(balIdList), len(self.EXP_BAL_ID_LIST));
        for sRow, eRow in zip(balIdList, self.EXP_BAL_ID_LIST):
            self.assertEqual(sRow, eRow);
    
    
    def test_main(self):
        """test the main function of BalanceIdFile"""
        instance = BalanceIdFile();
        
        args = ['', self.IN_FILENAME, self.OUT_FILENAME]
        instance.main(args)
        
        ifs = open(self.OUT_FILENAME)
        balIdList = [];
        for line in ifs:
            row = [int(x) for x in line.strip().split()]
            balIdList.append(row)
        ifs.close();
        
        self.assertEqual(len(balIdList), len(self.EXP_BAL_ID_LIST));
        for sRow, eRow in zip(balIdList, self.EXP_BAL_ID_LIST):
            self.assertEqual(sRow, eRow);
    


def suite():
    """Returns the suite of tests to run for this test class / module.
    Use unittest.makeSuite methods which simply extracts all of the
    methods for the given class whose name starts with "test"

    Actually, since this is mostly all basic input / output functions,
    can do most of it with doctests and DocTestSuite
    """
    suite = unittest.TestSuite();
    suite.addTest(unittest.makeSuite(TestBalanceIdFile));
    return suite;

if __name__=="__main__":
    unittest.TextTestRunner(verbosity=Const.RUNNER_VERBOSITY).run(suite())
