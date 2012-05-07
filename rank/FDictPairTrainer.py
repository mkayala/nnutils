#!/usr/bin/env python
# encoding: utf-8
"""
Given OP feature dictionary and data mapping to ordered pairs, train the shared weight network models. 

FDictPairTrainer.py
Created by Matt Kayala on 2010-10-04.
"""
import sys;
import os;
import gzip;
from optparse import OptionParser;
from pprint import pformat;

from nnutils.Util import FeatureDictReader;
from nnutils.mutil.MonteArchModel import MonteArchModel, loadArchModel, saveArchModel;
from nnutils.mutil.PairMonteFeatDictClassifier import PairMonteFeatDictClassifier;
from nnutils.mutil.Util import accuracy, rmse, sigmoid;
from nnutils.mutil.Const import EPSILON;

from numpy import array, zeros, concatenate, min, max, where;

from Util import log;

class FDictPairTrainer:
    """Provide a command line interface to running pairwise fdict training"""
    def __init__(self):
        """Constructor"""
        self.saveFile = True;
        self.archModelOutFile = None;
        self.fDictList = None;
        self.probArr = None;
        self.archModel = None;
        self.classifier = None;
        
    def main(self, argv):
        """Callable from Command line"""
        if argv is None:
            argv = sys.argv
        
        usageStr = \
            """usage: %prog [options] archModelInFile inData inProbArr archModelOutFile
            
            archModelInFile - pickled MonteArchModel file with the machine setup
            inData - zipped FeatureDict file
            inProbArr - space delim file with (lIdx, lDbId, rIdx, rDbId) per line
            archModelOutFile - filename to place final (and intermediate) trained model results
            """
        
        parser = OptionParser(usage = usageStr);
        parser.add_option('--nosaveint', dest='saveFile', action='store_false', default=True,
            help='If set, don\'t save the latest archmodel at the end of every epoch.')
        (options, args) = parser.parse_args(argv[1:])
        
        if len(args) == 4:
            self.options = options;
            self.saveFile = self.options.saveFile;
            
            archModelInFile = args[0]
            inDataFile = args[1]
            inProbArrFile = args[2]
            self.archModelOutFile = args[3];
            
            self.setup(inDataFile, inProbArrFile, archModelInFile);
            
            # Run some pre training stats
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
        lOut = theOut[self.probArr[:, 0]]
        rOut = theOut[self.probArr[:, 1]]
        sigOut = sigmoid(lOut - rOut)
        theAcc = accuracy(sigOut, 1)
        theRMSE = rmse(sigOut, 1)
        log.info('theAcc : %.4f, theRMSE : %.4f' % (theAcc, theRMSE))
    
    def setup(self, inDataFile, inProbArrFile, archModelInFile):
        """Method to load in the data and process the target data"""
        self.archModel = loadArchModel(archModelInFile);
        self.archModel.setupParams();
        
        ifs = open(inProbArrFile);
        self.probArr = [];
        for line in ifs:
            chunks = line.strip().split()
            self.probArr.append([int(chunks[0]), int(chunks[2])]);
        ifs.close();
        self.probArr = array(self.probArr);
        
        self.fDictList = [];
        ifs = gzip.open(inDataFile);
        reader = FeatureDictReader(ifs);
        for d in reader:
            self.fDictList.append(d);
        ifs.close();
        
        self.classifier = PairMonteFeatDictClassifier(self.archModel, self.fDictList,
                                                      self.probArr, self.postEpochCallback)
        self.classifier.setupModels();
    
    def postEpochCallback(self, classifier):
        """Callback method for the end of every epoch."""
        if self.saveFile:
            self.classifier.archModel.costTrajectory = self.classifier.costTrajectory;
            saveArchModel(self.classifier.archModel, self.archModelOutFile);

if __name__ == '__main__':
    instance = FDictPairTrainer();
    sys.exit(instance.main(sys.argv));
