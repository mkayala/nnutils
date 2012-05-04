#!/usr/bin/env python
# encoding: utf-8
"""
Given a trained shared weight network model and OP data in feat dict format, write out the predictions.

OPFDictPairPredictor.py

Created by Matt Kayala on 2010-10-04.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.
"""

import sys
import os
from optparse import OptionParser;

from Util import log;

class OPFDictPairPredictor:
    """Class to XX"""
    def __init__(self):
        """Constructor"""
        pass
    
    def main(self, argv):
        """Callable from Command line"""
        if argv is None:
            argv = sys.argv
        
        usageStr = \
            """usage: %prog [options] 
            """
        
        parser = OptionParser(usage = usageStr);
        (options, args) = parser.parse_args(argv[1:])
        
        if True:
            pass;
        else:
            parser.print_help();
            sys.exit(2);
    

if __name__ == '__main__':
    instance = OPFDictPairPredictor();
    sys.exit(instance.main(sys.argv));