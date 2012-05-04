#!/usr/bin/env python
# encoding: utf-8
"""
Class to train a simple classification model from orb pair feature dictionary data.

OPFDictClassTrainer.py

Created by Matt Kayala on 2010-10-04.
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

from CHEM.ML.orbDB.reactAtom.ReactAtomFDictTrainer import ReactAtomFDictTrainer;

from numpy import array, zeros, concatenate, min, max, where;

from Util import log;


class OPFDictClassTrainer(ReactAtomFDictTrainer):
    """Provide a cmd line interface to training a basic monte classification model with fdict data.
    
    This is very close to the ReactAtom version.  The only difference will be in how the data is setup.
    """
    def setup(self, inDataFile, inTargDataFile, archModelInFile):
        """Method to load in the data and process the target data"""
        self.archModel = loadArchModel(archModelInFile);
        self.archModel.setupParams();
        
        log.info('Reading in targFile: %s' % inTargDataFile)
        ifs = open(inTargDataFile)
        reader = csv.reader(ifs, delimiter=' ', quoting=csv.QUOTE_NONE);
        self.idxArr = [];
        self.targArr = [];
        for iRow, row in enumerate(reader):
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
    

if __name__ == '__main__':
    instance = OPFDictClassTrainer();
    sys.exit(instance.main(sys.argv));