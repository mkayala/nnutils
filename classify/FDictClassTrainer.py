#!/usr/bin/env python
# encoding: utf-8
"""
Command line interface to train a reactive atom prediction model using feature dictionaries.

ReactAtomFDictTrainer.py

Created by Matt Kayala on 2010-06-18.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.
"""

import sys
import os
import cPickle;
import gzip;
import csv;
from optparse import OptionParser;
from pprint import pformat;

from CHEM.ML.Util import FeatureDictReader;

from CHEM.ML.monteutils.MonteArchModel import MonteArchModel, loadArchModel, saveArchModel;
from CHEM.ML.monteutils.MonteFeatDictClassifier import MonteFeatDictClassifier;
from CHEM.ML.monteutils.Util import accuracy, rmse;
from CHEM.ML.monteutils.Const import EPSILON, MEMMAP_DTYPE;

from numpy import array, zeros, concatenate, min, max, where;


from Util import log;

class ReactAtomFDictTrainer:
    """Class to provide a command line interface to running react atom classification training
    over feature dictionary files"""
    def __init__(self):
        """Constructor"""
        self.saveFile = True;
        self.targcolumn = 1;
        self.archModelOutFile = None;
        self.fDictList = None;
        self.targArr = None;    # Actual targets
        self.idxArr = None;     # Map the targets to places in the featDictList
        #self.metaData = None;   # In memory copy of the csv idx file
        self.archModel = None;
        self.classifier = None;
    
    
    def main(self, argv):
        """Callable from Command line"""
        if argv is None:
            argv = sys.argv
        
        usageStr = \
            """usage: %prog [options] archModelInFile inData inTarg archModelOutFile 
            
            archModelInFile - MonteArchModel as created with MakeMonteArchModel.py with the 
                details of the architecture.  
            inData - a gzipped FeatureDict file
            inTarg - a text file with 'targcolumn' indicates which is the target to train to.
                Fmt: (idx, originalId, [target1, target2])
            archModelOutFile - Where to save the trained (and intermediate) models
            """
        
        parser = OptionParser(usage = usageStr);
        parser.add_option('--targcolumn', dest='targcolumn', type='int', default=2,
            help="Set which column of the target file is the actual target. (default=%default)")
        parser.add_option('--nosaveint', dest='saveFile', action='store_false', default=True,
            help='If set, don\'t save the latest archmodel at the end of every epoch.')
        (options, args) = parser.parse_args(argv[1:])
        
        if len(args) == 4:
            self.options = options;
            self.saveFile = self.options.saveFile;
            self.targcolumn = self.options.targcolumn;
            
            archModelInFile = args[0]
            inDataFile = args[1]
            inTargDataFile = args[2]
            self.archModelOutFile = args[3];
            
            self.setup(inDataFile, inTargDataFile, archModelInFile);
            
            # PreStats on the data self.classifier
            self.classifier.postEpochCall(-1)
            
            self.classifier.train()
            
            saveArchModel(self.classifier.archModel, self.archModelOutFile);
            
            self.runStats();
        else:
            parser.print_help();
            sys.exit(2);
    
    
    def runStats(self):
        """Convenience method to print out some stats about the training."""
        # And log some stats about accuracy, etc.
        theOut = self.classifier.apply(self.fDictList)
        theOut = theOut[self.idxArr];
        theAcc = accuracy(theOut, self.targArr)
        theRMSE = rmse(theOut, self.targArr)
        log.info('theAcc : %.4f, theRMSE : %.4f' % (theAcc, theRMSE))
    
    
    def setup(self, inDataFile, inTargDataFile, archModelInFile):
        """Method to load in the data and process the target data"""
        self.archModel = loadArchModel(archModelInFile);
        self.archModel.setupParams();
        
        ifs = open(inTargDataFile)
        #reader = csv.reader(ifs, quoting=csv.QUOTE_NONE);
        self.idxArr = [];
        self.targArr = [];
        for iRow, line in enumerate(ifs):
            row = line.strip().split();
            self.idxArr.append(int(row[0]));
            self.targArr.append(float(row[self.targcolumn]));
        ifs.close();
        self.idxArr = array(self.idxArr);
        self.targArr = array(self.targArr);
        
        
        log.info('Head(self.idxArr) : %s, tail(self.idxArr) : %s' % (pformat(self.idxArr[:5]), pformat(self.idxArr[-5:])))
        log.info('Head(self.targArr) : %s, tail(self.targArr) : %s' % (pformat(self.targArr[:5]), pformat(self.targArr[-5:])))
        
        self.fDictList = [];
        ifs = gzip.open(inDataFile);
        reader = FeatureDictReader(ifs);
        for d in reader:
            self.fDictList.append(d);
        ifs.close();
        log.info('FinalFDict data is %s ' % pformat(self.fDictList[-1]));
        
        self.classifier = MonteFeatDictClassifier(self.archModel, self.fDictList, self.targArr, self.idxArr, self.postEpochCallback)
        self.classifier.setupModels();
    
    
    def postEpochCallback(self, classifier):
        """Callback method for the end of every epoch."""
        if self.saveFile:
            self.classifier.archModel.costTrajectory = self.classifier.costTrajectory;
            saveArchModel(self.classifier.archModel, self.archModelOutFile);
        # No longer do this at the end of each epoch.  This is taken care of by the 
        # classifier itself/
        #self.runStats();
    
    


if __name__ == '__main__':
    instance = ReactAtomFDictTrainer();
    sys.exit(instance.main(sys.argv));