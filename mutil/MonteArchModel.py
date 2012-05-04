#!/usr/bin/env python
# encoding: utf-8
"""
A collection of Classes to encapsulate architecture models.  

Also some code to load and store the ArchModels.

MonteArchModel.py

Created by Matt Kayala on 2010-05-06.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.
"""

import sys
import os
from optparse import OptionParser, OptionGroup;
import cPickle;

from numpy.random import randn, random_sample;
from numpy import sqrt;

from MonteNeuralNetLayer import MonteNeuralNetClassifier;


class MonteArchModel:
    """Simple class to encapsulate some common parameters for learning with the monte pacakge"""
    def __init__(self):
        self.numhidden = 0;
        self.numfeats = 1;
        self.numparams = -1;
        
        self.trainertype = 'conjgrad'; #choices=['conjgrad', 'gdescmom', 'gdesc', 'bfgs']
        self.paramVar = 0.01;
        self.learningrate = 0.1;
        self.momentum = 0.01;
        self.l2decay = 0;
        self.numEpochs = 100;
        self.cgIterations = 1;
        self.gradientChunkSize = 500;
        self.onlineChunkSize = 2000;
        self.costEpsilon = 1E-6;
        self.batch = False;
        self.exponentAvgM = 0.95;
        self.qLearningRate = 0.05;
        
        self.params = None;
        self.costTrajectory = None;
        
        # Have a slot to store any parameters used in normalizing the data.
        self.normParams = None;
    
    
    def setupParams(self):
        """Convenience method to setup correct size parameters"""
        self.numparams = MonteNeuralNetClassifier.numparams(self.numfeats, self.numhidden);
        
        # Set these up in way that depends on the architecture
        self.params = 2.0 * random_sample(self.numparams) - 1;
        if self.numhidden == 0:
            self.params[:] *= self.paramVar / sqrt(self.numfeats + 1);
        else:
            self.params[:self.numfeats * self.numhidden + self.numhidden] *= self.paramVar / sqrt(self.numfeats + 1);
            self.params[self.numfeats * self.numhidden + self.numhidden: ] *= self.paramVar / sqrt(self.numhidden + 1);
    
    

def loadArchModel(fileName):
    """Convenience to load an arch model from a pickled file"""
    ifs = open(fileName);
    model = cPickle.load(ifs)
    ifs.close();
    return model;


def saveArchModel(model, fileName):
    """Convenience to write out an arch model to a pickled file"""
    ofs = open(fileName, 'w')
    cPickle.dump(model, ofs)
    ofs.close();


class MonteArchModelMaker:
    """Simple class that is extensible to be able to initialize a MonteArchModel"""
    def __init__(self):
        """Constructor, basically setup the option parser"""
        usageStr = \
            """usage: %prog [options] outFile;
            """
        
        self.parser = OptionParser(usage = usageStr);
        
        #Main architecture options
        self.parser.add_option('-n', '--numhidden', dest='numhidden', default=0, type='int',
                help='How many hidden nodes in the NN architecture.  Set to 0 for perceptron.  (Default %default)');
        self.parser.add_option('-f', '--numfeats', dest='numfeats', default=1, type='int',
                help='How many input features?')
        
        # Training options
        trainGroup = OptionGroup(self.parser, "Training Options",
                "Options relating to the training algorithms.")
        trainGroup.add_option('-t', '--trainertype', dest='trainertype', action='store', 
                type='choice', choices=['conjgrad', 'gdescmom', 'gdesc', 'bfgs', 'gdescadapt'], default='gdesc',
                help='What kind of optimization method should we choose? (default: %default)')
        trainGroup.add_option('--paramVar', dest='paramVar', default=0.1, type='float',
                help='Variance of the parameter initialization (normal with mean 0, default %default)')
        trainGroup.add_option('--learningrate', dest='learningrate', default=0.01, type='float',
                help='Stepsize (learning rate) for a gdesc flavor algorithm (default %default)');
        trainGroup.add_option('--momentum', dest='momentum', default=0.001, type='float',
                help='Momentum rate (for gdescmom algo default: %default)');
        trainGroup.add_option('--decay', dest='l2decay', default=0, type='float', 
                help='Weight decay parameter.  (Default %default)');
        trainGroup.add_option('--numEpochs', dest='numEpochs', default=100, type='int',
                help='Number of the epochs for training (Default %default)');
        trainGroup.add_option('--cgIterations', dest='cgIterations', type='int', default=1,
                help='Number of gradient calc iterations for each conj grad learning step.  (Default %default)');
        trainGroup.add_option('--gradientChunkSize', dest='gradientChunkSize', default=1000, type='int',
                help='Number of data points in each gradient calculation. (Default %default)');
        trainGroup.add_option('--onlineChunkSize', dest='onlineChunkSize', default=5000, type='int',
                help='Number of data points in each online learning step. (Default %default)');
        trainGroup.add_option('--costEpsilon', dest='costEpsilon', type='float', default=1E-6,
                help='Epsilon for determining convergence of algo. (default %default)');
        trainGroup.add_option('--batch', dest='batch', action='store_true', default=True)
        trainGroup.add_option('--online', dest='batch', action='store_false', default=True)
        trainGroup.add_option('--exponentAvgM', dest='exponentAvgM', type='float', default=0.95,
                help='Past weight exponential average parameter (Default %default)');
        trainGroup.add_option('--qLearningRate', dest='qLearningRate', type='float', default=0.05,
                help='Meta-learning rate (Default %default)');
        self.parser.add_option_group(trainGroup);
        
        self.filename = None;
    
    
    def createBasicArchModel(self, args, options):
        """Parse the options and make the basic feature file"""
        mArchModel = MonteArchModel();
        mArchModel.numhidden = options.numhidden;
        mArchModel.numfeats = options.numfeats;
        mArchModel.trainertype = options.trainertype;
        mArchModel.paramVar = options.paramVar;
        mArchModel.learningrate = options.learningrate;
        mArchModel.momentum = options.momentum;
        mArchModel.l2decay = options.l2decay;
        mArchModel.numEpochs = options.numEpochs;
        mArchModel.cgIterations = options.cgIterations;
        mArchModel.gradientChunkSize = options.gradientChunkSize;
        mArchModel.onlineChunkSize = options.onlineChunkSize;
        mArchModel.costEpsilon = options.costEpsilon;
        mArchModel.batch = options.batch;
        mArchModel.exponentAvgM = options.exponentAvgM;
        mArchModel.qLearningRate = options.qLearningRate;
        
        mArchModel.numparams = -1;
        mArchModel.setupParams();
        
        return mArchModel;
    
    
    def handleArgs(self, args):
        """Basic args handling. Can be overriden"""
        if len(args) == 1:
            self.fileName = args[0];
        else:
            self.parser.print_help();
            sys.exit(2);
    
    
    def handleOptions(self, options):
        """Basic method to handle doing anything with the options.  Can and maybe should be overridden"""
        pass;
    
    
    def main(self, argv):
        """Callable from Command line - Creates a pickled MonteArchModel"""
        if argv is None:
            argv = sys.argv
        
        (options, args) = self.parser.parse_args(argv[1:])
        
        self.handleArgs(args);
        self.handleOptions(options);
        mArchModel = self.createBasicArchModel(args, options)
        self.process(mArchModel, args, options);
    
    
    def process(self, mArchModel, args, options):
        """Main method to handle what to do with the built mArchModel"""
        saveArchModel(mArchModel, self.fileName);
    
    




if __name__ == '__main__':
    instance = MonteArchModelMaker();
    sys.exit(instance.main(sys.argv));
