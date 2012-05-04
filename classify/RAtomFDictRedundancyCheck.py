#!/usr/bin/env python
# encoding: utf-8
"""
Tool to check the RAtom Data for redundancy.  Are there any positives with exactly the same negatives.

RAtomFDictRedundancyCheck.py

Created by Matt Kayala on 2010-10-19.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.
"""

import sys
import os
import gzip
from optparse import OptionParser;

from CHEM.ML.Util import FeatureDictReader
from CHEM.Common.Util import ProgressDots;

from Util import log;

EPSILON = 1e-12;

class RAtomFDictRedundancyCheck:
    """Class to check for redundancy in the react atom data"""
    def __init__(self):
        """Constructor"""
        self.featDictList = [];
        self.featIdxByLength = {};
        self.posDataList = [];
        self.negDataByFDictIdx = {};
        self.column = 2;
        self.delimiter = ' '
    
    
    def main(self, argv):
        """Callable from Command line"""
        if argv is None:
            argv = sys.argv
        
        usageStr = \
            """usage: %prog [options] featDataFile classFile outFile
            """
        
        parser = OptionParser(usage = usageStr);
        parser.add_option('-c', '--column', dest='column', type='int', default=2);
        parser.add_option('-d', '--delimiter', dest='delimiter', default=' ');
        (options, args) = parser.parse_args(argv[1:])
        
        if len(args) == 3:
            self.featDataFile = args[0]
            self.classFile = args[1]
            self.outFile = args[2]
            self.column = options.column;
            self.delimiter = options.delimiter;
            
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
            for row in self.posDataList:
                idx = row[0]
                dbid = row[1]
                target = row[self.column]
                fd = self.featDictList[idx];
                for matchRow in self.findRedundancy(fd):
                    matchIdx = matchRow[0]
                    matchDbId = matchRow[1]
                    matchTarget = matchRow[self.column]
                    log.info('Match between %d, %d' % (dbid, matchDbId));
                    print >>ofs, 'Pos:', idx, dbid, target, 'Matches', matchIdx, matchDbId, matchTarget;
                progress.Update();
            progress.PrintStatus();
            ofs.close();
        
        else:
            parser.print_help();
            sys.exit(2);
    
    #
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
        progress = ProgressDots()
        for iLine, fd in enumerate(reader):
            self.featDictList.append(fd)
            theLength = len(fd.keys())
            if theLength not in self.featIdxByLength:
                self.featIdxByLength[theLength] = []
            self.featIdxByLength[theLength].append(iLine);
            progress.Update();
        
        progress.PrintStatus();
    
    
    def setupPosNegData(self, ifs):
        """Given an iterator over the class Data, setup the posDataList and negDataByFDictIdx.
        
        here have to be careful about repeated posData.  Also, need to split on self.delimiter 
        and cast things as ints.
        """
        self.posDataList = [];
        self.negDataByFDictIdx = {};
        
        log.info('About to load the class Data!!')
        progress = ProgressDots()
        posIdxSeenSet = set([])
        for line in ifs:
            row = [int(float(x))  for x in line.strip().split(self.delimiter)];
            if row[self.column] == 1:
                if row[0] not in posIdxSeenSet:
                    self.posDataList.append(row)
                    posIdxSeenSet.add(row[0]);
            else:
                if row[0] not in self.negDataByFDictIdx:
                    self.negDataByFDictIdx[row[0]] = []
                self.negDataByFDictIdx[row[0]].append(row);
            progress.Update();
        
        progress.PrintStatus()

if __name__ == '__main__':
    instance = RAtomFDictRedundancyCheck();
    sys.exit(instance.main(sys.argv));
