#!/usr/bin/env python
# encoding: utf-8
"""
Class to normalize a given feature dictionary from an existing calculated parameter set.

NormFeatDictFromParams.py

Created by Matt Kayala on 2011-03-13.
Copyright (c) 2011 Institute for Genomics and Bioinformatics. All rights reserved.
"""

import sys;
import os;
import gzip;
import cPickle;
from optparse import OptionParser;

from NormalizeFeatDicts import NormalizeFeatDicts;
from CHEM.ML.Util import FeatureDictReader, FeatureDictWriter;
from CHEM.Common.Util import ProgressDots;


from Util import log;
from Const import EPSILON;


class NormFeatDictFromParams:
    """Class to normalize a given feature dictionary from an existing calculated parameter set."""
    def __init__(self):
        """Constructor"""
        self.normObj = NormalizeFeatDicts();
    
    
    def loadNormParamObj(self, paramFile):
        """Convenience to load up the normParamObj from the paramFile"""
        ifs = open(paramFile);
        normParamObj = cPickle.load(ifs)
        ifs.close();
        return normParamObj;
    
    
    def main(self, argv):
        """Callable from Command line"""
        if argv is None:
            argv = sys.argv
        
        usageStr = \
            """usage: %prog [options] mapFile paramFile inFile outFile
            """
        
        parser = OptionParser(usage = usageStr);
        (options, args) = parser.parse_args(argv[1:])
        
        if len(args) == 4:
            mapFile, paramFile, inFile, outFile = args;
            
            self.normObj.loadFeatKeyToColMap(mapFile);
            normParamObj = self.loadNormParamObj(paramFile);
            
            log.info('Opening the raw feat dict %s' % inFile)
            ifs = gzip.open(inFile)
            reader = FeatureDictReader(ifs);
            
            log.info('Opening the output feat dict %s' % outFile)
            ofs = gzip.open(outFile, 'w')
            writer = FeatureDictWriter(ofs);
            
            progress = ProgressDots();
            for idx, fdict in self.normObj.normalizeFeatDictList(reader, normParamObj, self.normObj.featKeyToColMap):
                writer.update(fdict, str(idx));
                progress.Update();
            progress.PrintStatus();
            ifs.close();
            ofs.close();
        else:
            parser.print_help();
            sys.exit(2);
    

if __name__ == '__main__':
    instance = NormFeatDictFromParams();
    sys.exit(instance.main(sys.argv));