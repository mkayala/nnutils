#!/usr/bin/env python
# encoding: utf-8
"""
Class to gather some DB stats from the output of a redundancy check

RAtomRedundStats.py

Created by Matt Kayala on 2010-10-19.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.
"""

import sys
import os
from optparse import OptionParser;

from CHEM.Common.Util import stdOpen, ProgressDots;
from CHEM.Common.Env import SQL_PLACEHOLDER;
from CHEM.Common import DBUtil;

from CHEM.score.DB.Util import log, orbConnFactory;

from Util import log;

class RAtomRedundStats:
    """Class to gather some DB stats from the output of a redundancy check"""
    def __init__(self, connFactory=orbConnFactory):
        """Constructor"""
        self.connFactory = connFactory;
    
    
    def main(self, argv):
        """Callable from Command line"""
        if argv is None:
            argv = sys.argv
        
        usageStr = \
            """usage: %prog [options] inFile outFile
            """
        
        parser = OptionParser(usage = usageStr);
        (options, args) = parser.parse_args(argv[1:])
        
        if len(args) == 2:
            inFile = args[0]
            outFile = args[1]
            
            ifs = open(inFile)
            ofs = open(outFile, 'w')
            self.conn = self.connFactory.connection();
            
            progress = ProgressDots();
            posIdDict = {};
            for posId, negId in self.readLines(ifs):
                if posId not in posIdDict:
                    posData = self.lookupData(posId);
                    posIdDict[posId] = posData;
                posData = posIdDict[posId]
                negData = self.lookupData(negId)
                print >> ofs, posId, posData[0], posData[1], negId, negData[0], negData[1];
                progress.Update();
            progress.PrintStatus();
            
            self.conn.close();
            ifs.close()
            ofs.close()
            
        else:
            parser.print_help();
            sys.exit(2);
    
    
    def lookupData(self, atomId):
        """Given a single atom id, look up some info about the testcaseid, oprid"""
        qry = """SELECT test_case_id, orb_pair_reactants_id FROM atom WHERE atom_id=%s""" % SQL_PLACEHOLDER;
        
        res = DBUtil.execute(qry, (atomId,), conn=self.conn, connFactory=self.connFactory);
        return res[0];
    
    
    def readLines(self, ifs):
        """Read in a line and yield the relevant atom ids.
        
        NOTE: format of the data is something like:
        Pos: 68 386 1 Matches 5046 3298 0
        """
        for line in ifs:
            chunks = line.strip().split();
            id1 = int(chunks[2])
            id2 = int(chunks[6])
            yield (id1, id2);
    
    

if __name__ == '__main__':
    instance = RAtomRedundStats();
    sys.exit(instance.main(sys.argv));