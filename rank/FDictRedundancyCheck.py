#!/usr/bin/env python
# encoding: utf-8
"""
Class to check for redundancy in the fDict data. Using the class data, look at 
all the positives vs all the negatives.

Write out to the ofs file if there is redundancy.
FDictRedundancyCheck.py

Created by Matt Kayala on 2010-10-18.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.
"""

import sys
import os
import gzip
from optparse import OptionParser;

from CHEM.ML.Util import FeatureDictReader
from Util import log, ProgressDots;
from Const import PROG_BIG, PROG_SMALL;

EPSILON = 1e-8;

class FDictRedundancyCheck:
    """Class to check for redundancy in the fDict Data"""
    def __init__(self):
        """Constructor"""
        self.featDictList = [];
        self.featIdxByLength = {};
        self.posDataList = [];
        self.negDataByFDictIdx = {};
    
    
    def main(self, argv):
        """Callable from Command line"""
        if argv is None:
            argv = sys.argv
        
        usageStr = \
            """usage: %prog [options] featDataFile classFile outFile
            """
        
        parser = OptionParser(usage = usageStr);
        (options, args) = parser.parse_args(argv[1:])
        
        if len(args) == 3:
            self.featDataFile = args[0]
            self.classFile = args[1]
            self.outFile = args[2]
            
            ifs = gzip.open(self.featDataFile)
            reader = FeatureDictReader(ifs);
            self.setupFeatData(reader)
            ifs.close();
            
            ifs = open(self.classFile)
            self.setupPosNegData(ifs)
            ifs.close();
            
            log.info('Loaded everything, checking %d positives' % len(self.posDataList));
            ofs = open(self.outFile, 'w')
            progress = ProgressDots()
            for idx, dbid, target in self.posDataList:
                fd = self.featDictList[idx];
                for matchIdx, matchDbId, matchTarget in self.findRedundancy(fd):
                    log.info('Match between %d, %d' % (dbid, matchDbId));
                    print >>ofs, 'Pos:', idx, dbid, target, 'Matches', matchIdx, matchDbId, matchTarget;
                progress.Update();
            progress.PrintStatus();
            ofs.close();
        
        else:
            parser.print_help();
            sys.exit(2);
    
    
    def findRedundancy(self, fd):
        """Given a positive feature dict, find all redundant negative fds"""
        theLength = len(fd.keys());
        featIdxToCheck = self.featIdxByLength[theLength];
        
        for idx in featIdxToCheck:
            # Check equivalency
            if self.isEquivalentFeatureDict(fd, self.featDictList[idx]):
                # but then check to make sure is in the negIdxToFeatDictIdx
                if idx in self.negDataByFDictIdx:
                    for row in self.negDataByFDictIdx[idx]:
                        yield row;
    
    
    def isEquivalentFeatureDict(self, fd1, fd2):
        """Simple test of equivalence btwn two feat dicts"""
        keys1 = fd1.keys();
        keys2 = fd2.keys();
        
        keys1.sort();
        keys2.sort();
        
        if keys1 != keys2:
            return False;
        
        for key in keys1:
            if abs(fd1[key] - fd2[key]) > EPSILON:
                return False;
        
        return True;
    
    
    def setupFeatData(self, reader):
        """Setup any data structures for quickly determining redundancy.  Here will read in the fdicts, and make two setups
        First, one will be keyed solely by index, then the second will be a dict of the fdict lenghts with lists of their 
        fdictList idxs for each """
        self.featDictList = [];
        self.featIdxByLength = {};
        
        log.info('About to load featdict data!')
        progress = ProgressDots(PROG_BIG, PROG_SMALL)
        for iLine, fd in enumerate(reader):
            self.featDictList.append(fd)
            theLength = len(fd.keys())
            if theLength not in self.featIdxByLength:
                self.featIdxByLength[theLength] = []
            self.featIdxByLength[theLength].append(iLine);
            progress.Update();
        
        progress.PrintStatus();
    
    
    def setupPosNegData(self, ifs):
        """Given an iterator over the class Data, setup the posDataList and negDataByFDictIdx"""
        self.posDataList = [];
        self.negDataByFDictIdx = {};
        
        log.info('About to load the class Data!!')
        progress = ProgressDots(PROG_BIG, PROG_SMALL)
        for line in ifs:
            row = [int(x)  for x in line.strip().split()];
            if row[-1] == 1:
                self.posDataList.append(row)
            else:
                if row[0] not in self.negDataByFDictIdx:
                    self.negDataByFDictIdx[row[0]] = []
                self.negDataByFDictIdx[row[0]].append(row);
            progress.Update();
        
        progress.PrintStatus()
        
        
        

if __name__ == '__main__':
    instance = FDictRedundancyCheck();
    sys.exit(instance.main(sys.argv));