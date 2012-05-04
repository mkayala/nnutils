#!/usr/bin/env python
# encoding: utf-8
"""
Utilities module for NNUtils.

Basic file->dictionary representations

File: Util.py
Author: MKayala
Created 2012-05-04
Copyright 2012. 

Miscellaneous utility functions used across the application.

FeatureDict Reader and Writer allows for the saving and loading
of feature dictionary data generated from feature extractors to 
plain text file formats.

Largely taken from Baldi Group CHEM Repository.  2012.
"""
import sys, os;
import time

import Const
from Const import FEATURE_PREFIX, KEY_DELIM, TEXT_DELIM;

class FeatureDictWriter(dict):
    """Utility class to encode data feature vectors 
    (represented as string:value feature dictionaries) into a plain text file format.
    
    These should then be re-read using the matching "FeatureDictReader" class.
    The basic strategy is to identify every feature encountered among the data 
    items and assign each a unique index number.  This class 
    (which extends the dict class) stores the feature:index mappings and then 
    prints out each data item with the corresponding index:value pairs.
    """

    def __init__(self, outfile=None):
        """Constructor.  Just pass it the output file 
        (object, not filename) to write to.
        """
        dict.__init__(self);    # Super-class constructor
        self.featureCount = 0;
        self.outfile = outfile;

    def __getitem__(self, feature):
        """Override dictionary access method "dict[key]" to 
        automatically add items for newly encountered keys.
        """
        try:
            return dict.__getitem__(self, feature)
        except KeyError:
            self.add(feature)
            return dict.__getitem__(self, feature)

    def add(self, feature):
        """Makes the writer aware of a feature that will have to be 
        written for subsequent feature dictionaries.
        
        Should be called for every possible feature in the dataset before 
        actually trying to write the data to the output file with the 
        "update" method.  This way the object can first assign index 
        numbers to every feature.
        
        Return value indicates whether the feature is new to the writer or not
        """
        if feature not in self:
            self[feature] = self.featureCount    # Generate and record the next index number for this new feature
            self.featureCount += 1;
            return True
        return False

    def __setitem__(self, feature, featureIndex):
        """Override dictionary set method "dict[key] = value" 
        to automatically address newly encountered features.
        """
        if feature not in self:
            self.new_key(feature, featureIndex);
        dict.__setitem__(self, feature, featureIndex); # Super-class implementation

    def formatFeature(feature):
        """Return a string representation of a feature suitable for storage in the file.
        
        Should be structured enough to be parseable back into object form by a
        respective FeatureDictReader.parseFeature method.
        By default will just use the "__str__" interface to format it.
        
        For something more sophisticated, you should create your own FeatureDictWriter
        sub-class that overrides this method.  You can then write a respective
        extension to the FeatureDictReader to parse the string back into an object.
        """
        return str(feature);
    formatFeature = staticmethod(formatFeature);

    def new_key(self, feature, featureIndex):
        """Output the given feature:index mapping.  
        
        Automatically invoked by calls to the set and "add" methods for newly encountered features.
        """
        if self.outfile is not None: 
            # Output format:    "#  %(index)s   %(feature)s"
            self.outfile.write(FEATURE_PREFIX);
            self.outfile.write(TEXT_DELIM);
            self.outfile.write(str(featureIndex));
            self.outfile.write(TEXT_DELIM);
            self.outfile.write(self.formatFeature(feature));
            print >> self.outfile;  # New line

    def update(self, featureDict, description, nameID="UNKNOWN_ID"):
        """Output a specific feature dictionary to text format.

        It would be nice to call the "add" method on every possible feature
        before calling this method so that the feature:index mappings will all be
        output at the beginning of the file, instead of intermixed with the data.
        However, this method will automatically try doing so if it has not.
        Either way, guaranteed that each feature:index mapping will appear before
        they are ever referenced by a data row.
        
        The provided object's description will be printed first for each.  
        It is important that this description NOT: 
            - be empty or
            - contain any whitespace or 
            - equal the FEATURE_PREFIX "#"
        Otherwise the "decoding" steps later will be confused.  
        
        Preferably this description should be some kind of data identifying string, 
        but uniqueness is not enforced.
        """
        for feature in featureDict.iterkeys():
            self.add(feature)   # If it was added before, will automatically be ignored

        if self.outfile is not None:
            # Translate feature:value mappings into index:value mappings
            featureValueMap = [];
            for feature, value in featureDict.iteritems():   
                featureIndex = self[feature];
                featureValueMap.append( (featureIndex, value) );
            featureValueMap.sort(); # Nicer to print the features in a consistent order
            
            # Output format:    "%(description)s    %(non_zero_feat_j)d:%(val_j)d   %(non_zero_feat_j+1)d:%(val_j+1)d ..."
            self.outfile.write(description.strip());
            self.outfile.write(TEXT_DELIM);
            if nameID == '':
                self.outfile.write("UNKNOWN_ID")
            else:
                self.outfile.write(nameID.strip());
            self.outfile.write(TEXT_DELIM);
            for featureIndex, value in featureValueMap:
                self.outfile.write(str(featureIndex));
                self.outfile.write(KEY_DELIM);
                self.outfile.write(str(value));
                self.outfile.write(TEXT_DELIM);
            print >> self.outfile # New line

class FeatureDictReader(dict):
    """Decodes data feature vectors (represented as index:value lines in a text file) 
    into string:value feature dictionary objects.
    
    These were probably encoded in the first place with the matching 
    "FeatureDictWriter" class.  The basic strategy is to just back-translate the 
    index numbers into the actual string representation features.  
    This class (which extends the dict class) stores the index:feature mappings 
    for subsequent access  (note that this is the inverse of the 
    FeatureDictWriter's feature:index mappings).
    
    Note that the default implementation assumes the features are represented 
    as simple text strings.  If you want a more sophisticated object representation, 
    you'll have to extend this class and override the parseFeature method to 
    translate the string into the feature object.

    Example Usage:
    (Note this may have problems as a doctest since the feature:index mapping order is arbitrary
    based on the "random" traversal of feature keys through the feature dictionaries.

    >>> from cStringIO import StringIO
    >>> infile = StringIO();        # doctest can't handle multi-line strings well
    >>> print >> infile, ""         # Test blank line robustness
    >>> print >> infile, "# 0 a"    # So write it out as a StringIO first
    >>> print >> infile, "# 1 s"
    >>> print >> infile, "# 2 d"
    >>> print >> infile, "# 3 g"
    >>> print >> infile, "# 4 f"
    >>> print >> infile, "# 5 A"
    >>> print >> infile, "# 6 F"
    >>> print >> infile, "# 7 S"
    >>> print >> infile, "# 8 D"
    >>> print >> infile, "# 9 h"
    >>> print >> infile, "# 10 G"
    >>> print >> infile, "# 11 H"
    >>> print >> infile, "asdfsdfg UNKNOWN_ID 0:1 1:2 2:2 3:1 4:2 "
    >>> print >> infile, "asdfasdfASDF UNKNOWN_ID 0:2 1:2 2:2 4:2 5:1 6:1 7:1 8:1 "
    >>> print >> infile, "dfghDFGH UNKNOWN_ID 2:1 3:1 4:1 6:1 8:1 9:1 10:1 11:1 "
    >>> infile = StringIO(infile.getvalue());
    >>>
    >>> featureReader = FeatureDictReader(infile);
    >>> # Read out and print the contents of each feature dictionary
    >>> for featureDict in featureReader:
    ...     print str(featureDict)
    {'a': 1.0, 's': 2.0, 'd': 2.0, 'g': 1.0, 'f': 2.0}
    {'a': 2.0, 'A': 1.0, 'd': 2.0, 'F': 1.0, 'f': 2.0, 'S': 1.0, 's': 2.0, 'D': 1.0}
    {'D': 1.0, 'g': 1.0, 'F': 1.0, 'h': 1.0, 'f': 1.0, 'G': 1.0, 'H': 1.0, 'd': 1.0}
    >>> for description in featureReader.objDescriptions:
    ...     print description
    asdfsdfg
    asdfasdfASDF
    dfghDFGH
    """

    """File object to read from"""
    infile = None;
    
    """Object descriptions read out from the file, should be in position sync 
    with the feature dictionaries that come out of the iterator.
    """
    objDescriptions = None;
    objNameIDs = None;

    def __init__(self, infile):
        """Constructor expects an input file (object, not filename) to read from."""
        dict.__init__(self);    # Super-class constructor;
        self.infile = infile;
    
    def parseFeature(featureStr):
        """Given a string representation of the feature, extracted from the 
        feature file, return the actual feature object to key the dictionaries by.  
        For string based kernels (and by default), this can just be the string itself.
        """
        return featureStr;
    parseFeature = staticmethod(parseFeature);
    
    def __iter__(self):
        """Produce an iterator over the feature dictionary objects parsed out of the input file.

        This method implements the __iter__ interface which means you can do something as simple as:
        
        >>> from cStringIO import StringIO
        >>> reader = FeatureDictReader(StringIO(""))
        >>> for (featureDict, objDescr) in reader:
        ...     print objDescr, featureDict

        However, this overrides the normal meaning of the dictionary __iter__ method.
        Normally the __iter__ method should get the feature *keys* stored in the 
        reader dictionary rather than data pairs.  To access the keys in this way, 
        they must instead be accessed explicitly with the iterkeys() method.

        This method can only be called once as it iterates through the 
        source file, after which time there's no guarantee we can trace back to 
        the start of the file.  If you do wish to be able to produce multiple 
        iterators over the same data, use the FeatureDictReaderFactory which will 
        create a temp file as needed to generate as many FeatureDictReader 
        iterators as requested.
        """
        self.objDescriptions = [];
        self.objNameIDs = [];
        
        for line in self.infile:
            tokens = line.split();
            if len(tokens) < 1: continue;   # Blank line, just skip it

            if tokens[0] == FEATURE_PREFIX:
                # Extract the feature:index mapping from this line
                #prefix = tokens[0];
                featureIndex = tokens[1];
                feature = self.parseFeature(tokens[2]);
                self[featureIndex] = feature;
            else:
                # Assume this is a data row.  Decode it back into the original feature dictionary format
                description = tokens[0];
                featureDict = dict();
                nameID = tokens[1];
                for featureIndexPair in tokens[2:]:   # Everything but first token should be an index:value mapping
                    (featureIndex, value) = featureIndexPair.split(KEY_DELIM);
                    #index  = int(index);   # Unneeded since the feature:index mappings were left as strings too
                    value   = float(value); # Assume is a float
                    feature = self[featureIndex];
                    featureDict[feature] = value;
                
                self.objDescriptions.append(description);
                self.objNameIDs.append(nameID);
                yield featureDict;   # Using Python "generator" pattern for an easy iterator
    
    def featureNameIterator(self):
        """Iterator over just the feature names (to allow simple building of column maps)"""
        for line in self.infile:
            tokens = line.split();
            if len(tokens) < 1: continue;   # Blank line, just skip it

            if tokens[0] == FEATURE_PREFIX:
                # Extract the feature:index mapping from this line
                #prefix = tokens[0];
                featureIndex = tokens[1];
                feature = self.parseFeature(tokens[2]);
                self[featureIndex] = feature;
                yield feature;
    
    
class ProgressDots:
    """Clone of Openeyes OEChem OEDots class, to add progress indicator to long processes,
    without actually requiring OEChem as a dependency.
    """
    def __init__( self, big=Const.PROG_BIG, small=Const.PROG_SMALL, name="items", stream=sys.stderr ):
        """Constructor.
        big - Number of updates before completing a progress output line.
        small - Number of updates before outputting a progress dot.
        name - Name of the items being processed.
        stream - Stream to send progress output to.  Defaults to sys.stderr.
        """
        self.big = big;
        self.small = small;
        self.name = name;
        self.stream = sys.stderr;
        self.count = 0;
        self.start = time.time();

    def Update(self,step=1):
        """Update the progress counter by an increment of size step (default=1).
        Well output progress dots or line information to the stream if
        reached an appropriate big or small increment.
        """
        self.count += step;
        if self.small > 0 and self.count % self.small == 0:
            self.stream.write(".");
        if self.big > 0 and self.count % self.big == 0:
            self.PrintStatus()
    
    def GetCounts(self):
        """Get the current count of updates"""
        return self.count;
    
    def GetTime(self):
        """Get the time (in seconds) since the progressindicator was created."""
        return time.time()-self.start;

    def PrintStatus(self):
        # Assume 0 values means no feedback desired
        if self.small > 0 and self.big > 0:
            print >> self.stream, "%d %s processed after %d seconds." % (self.count,self.name,self.GetTime());

