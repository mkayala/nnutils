#!/usr/bin/env python
# encoding: utf-8
"""
Class to normalize a set of feature dictionaries to have ~ 0-1 scale.

NormalizeFeatDicts.py

Created by Matt Kayala on 2010-10-27.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.
"""

import sys;
import os;
import gzip;
import cPickle;
from optparse import OptionParser;

from Util import FeatureDictReader, FeatureDictWriter;
from Util import ProgressDots;

from Util import log;
from Const import EPSILON;

class NormalizeFeatDicts:
    """Class to Normalize a gzipped feature dict file."""
    def __init__(self, featKeyToColMap=None, logwarning=False):
        """Constructor"""
        self.featKeyToColMap = featKeyToColMap;
        self.logwarning = logwarning
    
    
    def loadFeatKeyToColMap(self, pickleFile):
        """Given a name of a pickled file.., setup the keyToColMap"""
        ifs = open(pickleFile)
        self.featKeyToColMap = cPickle.load(ifs)
        ifs.close();
    
    
    def processFeatureDictList(self, featureReader):
        """Given a feature dict reader, go through and calculate a minMax dict for each feature."""
        log.info('Beginning processing a featDict');
        minMaxDict = {};
        idList = [];
        fDictList = [];
        progress = ProgressDots();
        for iRow, fDict in enumerate(featureReader):
            minMaxDict = self.updateMinMaxDict(minMaxDict, fDict);
            progress.Update();

        progress.PrintStatus();
        
        minRangeDict = {};
        minRangeDict = self.convertMinMaxDictToMinRange(minMaxDict)
        return minRangeDict;
    
    
    def updateMinMaxDict(self, minMaxDict, fDict):
        """This will update a dictionary of the minimum and maximum for each key"""
        for key, val in fDict.iteritems():
            if key in minMaxDict:
                # Is new max?
                if val > minMaxDict[key][1]:
                    minMaxDict[key][1] = val;
                # Is new min?
                if val < minMaxDict[key][0]:
                    minMaxDict[key][0] = val;
            else:
                minMaxDict[key] = [val, val];
        return minMaxDict;
    
    
    def convertMinMaxDictToMinRange(self, minMaxDict):
        """Convert a dictionary with the min and max for each feature into a set with tuples of min and range
        
        Adding in a correction here, where if we have a minVal less than 10, set the minVal to really be min(minVal, 0)
        """
        newDict = {};
        for key, (minVal, maxVal) in minMaxDict.iteritems():
            if minVal < 100 or abs(maxVal-minVal) < EPSILON:
                minVal = min(minVal, 0)
            newDict[key] = [minVal, 1.0];
            if maxVal - minVal > 1.0:
                newDict[key] = [minVal, maxVal-minVal + EPSILON];
            
        return newDict;
    
    
    def saveNormalizationParams(self, normParamObj, pklFileName):
        """Method to save the normalization params"""
        ofs = open(pklFileName, 'w');
        cPickle.dump(normParamObj, ofs);
        ofs.close();
    
    
    def normalizeFeatDictList(self, featureReader, minMaxDict, colMapObj):
        """Given a fetaure reader.  A norm param dictionary, and a new colMapObj,
        yield out tuples of the form (id, newMappedFDict)"""
        for fDict in featureReader:
            newDict = {};
            idVal = featureReader.objDescriptions[-1];
            for key, val in fDict.iteritems():
                if key in minMaxDict and key in colMapObj:
                    (minVal, rangeVal) = minMaxDict[key]
                    newKey = colMapObj[key]
                    newDict[newKey] = (val - minVal)/rangeVal;
                else:
                    if key not in minMaxDict and self.logwarning:
                        log.warning('key : %s not in minMaxDict' % str(key))
                        pass;
                    if key not in colMapObj and self.logwarning:
                        log.warning('key : %s not in colMapObj' % str(key));
                        pass
            yield (idVal, newDict)
    
    
    def main(self, argv):
        """Callable from Command line"""
        if argv is None:
            argv = sys.argv
        
        usageStr = \
            """usage: %prog [options] mapFile inFile outFile
            """
        
        parser = OptionParser(usage = usageStr);
        parser.add_option('-p', '--pklfile', dest='pklfile', default=None, 
            help='Set to save the normalization params');
        parser.add_option('-t', '--testFile', dest='testFile', default=None,
            help='Set to load in this test file to normalize as well.')
        parser.add_option('-o', '--outTestFile', dest='outTestFile', default=None,
            help='Where to save the outTestFile')
        parser.add_option('-w', '--logwarning', dest='logwarning', default=False,
                          action='store_true',
                          help='Log warnings about features missing from maps/data')
        
        (options, args) = parser.parse_args(argv[1:])
        
        if len(args) == 3:
            mapFile = args[0];
            inFile = args[1];
            outFile = args[2];
            self.logwarning = options.logwarning
            
            self.loadFeatKeyToColMap(mapFile);
            
            # First read through the original file and calc the params
            log.info('Calculating the norm params on %s' % inFile)
            ifs = gzip.open(inFile)
            reader = FeatureDictReader(ifs);
            normParamsDict = self.processFeatureDictList(reader)
            ifs.close();
            
            #Save the params
            if options.pklfile is not None:
                self.saveNormalizationParams(normParamsDict, options.pklfile)
            
            # Then process the inFile
            log.info('About to normalize : %s' % outFile)
            ofs = gzip.open(outFile, 'w')
            writer = FeatureDictWriter(ofs);
            ifs = gzip.open(inFile)
            reader = FeatureDictReader(ifs)
            progress = ProgressDots();
            for idx, fDict in self.normalizeFeatDictList(reader, normParamsDict, self.featKeyToColMap):
                writer.update(fDict, str(idx));
                progress.Update();
            progress.PrintStatus();
            ifs.close();
            ofs.close();
            
            # then if the -t and -o options were set do the same on the test Data
            if options.testFile is not None and options.outTestFile is not None:
                ifs = gzip.open(options.testFile)
                ofs = gzip.open(options.outTestFile, 'w')
                reader = FeatureDictReader(ifs)
                writer = FeatureDictWriter(ofs)
                log.info('About normalize : %s' % options.testFile);
                progress = ProgressDots();
                for idx, fDict in self.normalizeFeatDictList(reader, normParamsDict, self.featKeyToColMap):
                    writer.update(fDict, str(idx));
                    progress.Update();
                progress.PrintStatus();
                ifs.close();
                ofs.close();
            
            
        else:
            parser.print_help();
            sys.exit(2);
    

if __name__ == '__main__':
    instance = NormalizeFeatDicts();
    sys.exit(instance.main(sys.argv));
