import Const
import sys, os
import logging
import cgi, UserDict
import unittest

log = logging.getLogger(Const.APPLICATION_NAME)
log.setLevel(Const.LOGGER_LEVEL)

handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter(Const.LOGGER_FORMAT)

handler.setFormatter(formatter)
log.addHandler(handler)

### A function for testing feature dicts:
def spectrumExtractor(obj, k=1):
    """Create a dictionary keyed by all the k-mers (k-length substrings)
    of the input string object, with values equal to the number of times 
    that k-mer appears in the string.
    """
    featureDict = {}
    aString = obj.strip()
    n = len(aString)
    for i in xrange(n - k + 1):
        substr = aString[ i : i+k ]
        if substr not in featureDict: # If the substr has not been found, create a new entry for it
            featureDict[substr] = 0
        featureDict[substr] += 1 # Increment the count for the found substr
    return featureDict

