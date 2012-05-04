#!/usr/bin/env python
# encoding: utf-8
"""
Class to balance an id file callable from the command line

BalanceIdFile.py

Created by Matt Kayala on 2010-09-17.
"""

import sys
import os
from optparse import OptionParser;

from nnutils.Util import ProgressDots;
from Util import log;

class BalanceIdFile(object):
    """Class to balance a file of ids"""
    def __init__(self):
        """Constructor"""
        pass
    
    def main(self, argv):
        """Callable from Command line"""
        if argv is None:
            argv = sys.argv
        
        usageStr = \
            """usage: %prog [options] inIdFile outIdFile
            
            Expect idFile to be of the format (idx dbIdx class)
            Will repeat lines of idFile in outIdFile to approximately balance the class distributions.
            """
        
        parser = OptionParser(usage = usageStr);
        parser.add_option('-c', '--column', dest='column', type='int', default=2)
        (options, args) = parser.parse_args(argv[1:])
        
        if len(args) == 2:
            inIdFile = args[0];
            outIdFile = args[1];
            column = options.column
            ifs = open(inIdFile)
            ofs = open(outIdFile, 'w');
            
            rawIdList = [];
            for line in ifs:
                line = line.strip();
                row = [int(x) for x in line.split()];
                rawIdList.append(row)
            
            for line in self.balanceIdList(rawIdList, column):
                print >>ofs, ' '.join([str(x) for x in line]);
            
            ifs.close();
            ofs.close();
            
        else:
            parser.print_help();
            sys.exit(2);
    
    
    def balanceIdList(self, idList, balanceByCol=2):
        """Given an IdList object, repeat the entries labeled with 1 so that we have close to the same number
        of entries with label 0."""
        numZeros = 0;
        numOnes = 0;
        
        for row in idList:
            if int(row[balanceByCol]) == 0:
                numZeros += 1;
            else:
                numOnes += 1;
        
        numTimesToRepeat = int(float(numZeros)/numOnes) - 1;
        ## Artifically up this a little bit.., 
        numTimesToRepeat += 3;
        log.info('In balance, will repeat %d times' % numTimesToRepeat);
        
        for row in idList:
            yield row;
            if int(row[balanceByCol]) != 0:
                for iRepeat in range(numTimesToRepeat):
                    yield row;
        
        return;
    
if __name__ == '__main__':
    instance = BalanceIdFile();
    sys.exit(instance.main(sys.argv));
