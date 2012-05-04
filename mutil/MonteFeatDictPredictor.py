#!/usr/bin/env python
# encoding: utf-8
"""
Class to run through a feat dict file and make predictions

MonteFeatDictPredictor.py

Created by Matt Kayala on 2010-10-10.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.
"""

import sys
import os
import gzip;
import csv;
from optparse import OptionParser;

from optparse import OptionParser;

from CHEM.Common.Util import ProgressDots;
from CHEM.ML.Util import FeatureDictReader;
from MonteArchModel import MonteArchModel, loadArchModel;
from MonteFeatDictClassifier import MonteFeatDictClassifier;
from Const import EPSILON;
from numpy import array, zeros, min, max, where, newaxis;

from Util import log;

class MonteFeatDictPredictor:
    """Class to make predictions on fDict format data given a trained model."""
    def __init__(self, archModel=None, chunkSize=500):
        """Constructor"""
        self.archModel = archModel;
        self.chunkSize = chunkSize;
    
    
    def setup(self):
        """Given that the self.archModel is correctly loaded, setup the machinery to run 
        the predictions"""
        self.classifier = MonteFeatDictClassifier(self.archModel)
        self.classifier.setupModels();
    
    
    def main(self, argv):
        """Callable from Command line"""
        if argv is None:
            argv = sys.argv
        
        usageStr = \
            """usage: %prog [options] archModelFile featDataFile idxArrFile outFile
            """
        parser = OptionParser(usage = usageStr);
        parser.add_option('-d', '--deduplicate', dest='deduplicate', action='store_true', default=False,
            help='Deduplicate the idArr');
        parser.add_option('--delim', dest='delim', default=' ', 
            help='Delimiter of the idxArrFile (default "%default").  Note outFile is always space-delim.')
        (options, args) = parser.parse_args(argv[1:])
        
        if len(args) == 4:
            (archModelFile, featDataFile, idxArrFile, outFile) = args;
            self.archModel = loadArchModel(archModelFile);
            self.setup();
            
            ifs = open(idxArrFile);
            reader = csv.reader(ifs, quoting=csv.QUOTE_NONE, delimiter=options.delim);
            seenIds = set([]);
            idArr = [];
            for row in reader:
                row[0] = int(row[0]);
                #row[1:] = [float(x) for x in row[1:]];
                if options.deduplicate:
                    if row[0] not in seenIds: 
                        idArr.append(row);
                        seenIds.add(row[0]);
                else:
                    idArr.append(row);
            #idArr = array(idArr);
            ifs.close();
            
            ifs = gzip.open(featDataFile)
            reader = FeatureDictReader(ifs);
            fDictList = [];
            for aDict in reader:
                fDictList.append(aDict);
            ifs.close();
            
            progress = ProgressDots();
            
            ofs = open(outFile, 'w');
            for idChunk, predValChunk in self.predict(fDictList, idArr):
                for iRow in range(len(idChunk)):
                    chunkTargData = [str(x) for x in idChunk[iRow]]
                    chunkTargData = ' '.join(chunkTargData)
                    print >>ofs, chunkTargData, predValChunk[iRow];
                    progress.Update();
            ofs.close();
            progress.PrintStatus();
        else:
            parser.print_help();
            sys.exit(2);
    
    
    def predict(self, fDictList, idArr):
        """Run the fDictList through the trained model in chunks"""
        for start in range(0, len(idArr), self.chunkSize):
            end = start + self.chunkSize;
            subFDictList = fDictList[start:end]
            subIdArr = idArr[start:end];
            
            predictions = self.classifier.apply(subFDictList);
            yield (subIdArr, predictions);
    
    

if __name__ == '__main__':
    instance = MonteFeatDictPredictor();
    sys.exit(instance.main(sys.argv));