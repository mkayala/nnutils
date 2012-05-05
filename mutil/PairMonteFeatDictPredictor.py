#!/usr/bin/env python
# encoding: utf-8
"""
Class to write out predictions for a Pairwise monte feat dict predictor, callable from Command Line.

PairMonteFeatDictPredictor.py
Created by Matt Kayala on 2010-10-18.

Has Command line interface
"""

import sys;
import os;
import gzip;
from optparse import OptionParser;

from nnutils.Util import ProgressDots;
from nnutils.Util import FeatureDictReader;
from MonteArchModel import MonteArchModel, loadArchModel;
from PairMonteFeatDictClassifier import PairMonteFeatDictClassifier;
from Const import EPSILON;
from numpy import array, zeros, min, max, where, newaxis;

from Util import log;

class PairMonteFeatDictPredictor:
    """Class to make predictions on fDict format data given a trained pairwise model"""
    def __init__(self, archModel=None, chunkSize=500):
        """Constructor"""
        self.archModel = archModel;
        self.chunkSize = chunkSize
    
    def loadArchModelFromFile(self, fileName):
        """Convenience to load up the archModel from a file"""
        self.archModel = loadArchModel(fileName);
    
    def setup(self):
        """Given that self.archModel is correctly loaded, setup machinery to run predictions"""
        self.predictor = PairMonteFeatDictClassifier(self.archModel)
        self.predictor.setupModels();
    
    
    def main(self, argv):
        """Callable from Command line"""
        if argv is None:
            argv = sys.argv
        
        usageStr = \
            """usage: %prog [options] archModelFile featDataFile outFile
            """
        
        parser = OptionParser(usage = usageStr);
        (options, args) = parser.parse_args(argv[1:])
        
        if len(args) == 3:
            archModelFile = args[0]
            featDataFile = args[1]
            outFile = args[2]
            
            #Read in the 
            self.archModel = loadArchModel(archModelFile);
            self.setup();
            
            # Then set up the reader and writer
            ifs = gzip.open(featDataFile);
            reader = FeatureDictReader(ifs);
            ofs = open(outFile, 'w');
            progress = ProgressDots();
            
            for value in self.predict(reader):
                print >> ofs, value;
                progress.Update();
            progress.PrintStatus();
            
            ifs.close();
            ofs.close();
            
        else:
            parser.print_help();
            sys.exit(2);
    
    
    def predict(self, reader):
        """Run through the data, predicting a value for each data point"""
        fDataList = [];
        for iLine, fd in enumerate(reader):
            fDataList.append(fd);
            if iLine % self.chunkSize == 0 and len(fDataList) > 0:
                for val in self.predictor.apply(fDataList):
                    yield val;
                fDataList = [];
        
        if len(fDataList) > 0:
            for val in self.predictor.apply(fDataList):
                yield val;

if __name__ == '__main__':
    instance = PairMonteFeatDictPredictor();
    sys.exit(instance.main(sys.argv));
