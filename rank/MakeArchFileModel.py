#!/usr/bin/env python
# encoding: utf-8
"""
Simple script to set up arch file models for a particular orb pair experiment.

MakeArchFileModel.py

Created by Matt Kayala on 2010-10-04.
"""
import sys
import os
from optparse import OptionParser;
import cPickle;

from nnutils.mutil.MonteArchModel import MonteArchModelMaker;

from Util import log;

class MakeArchFileModel(MonteArchModelMaker):
    """Class to handle initialization of an architecture file for an Orb Pair ML expt."""
    def __init__(self):
        """Constructor"""
        MonteArchModelMaker.__init__(self);
        usageStr = \
            """usage: %prog [options] mapFile archFileModel
            
            Given a map file from the binary data write (to determine the number of features) and a place to
            write out and some options build up a config pickled arch file for a monte training experiment.
            """
        
        self.parser.set_usage(usageStr);
        
        # Main thing we are grabbing here is the number of features.
        self.numfeats = None;
        self.mapObj = None;
    
    
    def createBasicArchModel(self, args, options):
        """Handle figuring out num feats etc."""
        mArchModel = MonteArchModelMaker.createBasicArchModel(self, args, options)
        
        self.numfeats = len(self.mapObj.keys());
                
        mArchModel.numfeats = self.numfeats;
        mArchModel.setupParams();
        return mArchModel;
    
    
    def handleOptions(self, options):
        """Basically read in the special features if any"""
        pass;
    
    
    def handleArgs(self, args):
        """Read in the mapfile to figure out the number of features"""
        if len(args) == 2:
            self.featFile = args[0];
            self.fileName = args[1];
            
            ifs = open(self.featFile)
            self.mapObj = cPickle.load(ifs);
            ifs.close();
        else:
            self.parser.print_help();
            sys.exit(2);
    
    

if __name__ == '__main__':
    instance = MakeArchFileModel();
    sys.exit(instance.main(sys.argv));
