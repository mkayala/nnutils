#!/usr/bin/env python
# encoding: utf-8
"""
Util.py

Created by Matt Kayala on 2010-05-04.
"""

import Const
import sys, os
import logging

log = logging.getLogger(Const.APPLICATION_NAME)
log.setLevel(Const.LOGGER_LEVEL)

handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter(Const.LOGGER_FORMAT)

handler.setFormatter(formatter)
log.addHandler(handler)


"""Here are some common ML functions."""
from numpy import exp;
def sigmoid(x):
    return 1./(1. + exp(-x));


def identity(x):
    return x;


def rmse(outputs, targets):
    """Simple function to calc rmse."""
    from numpy import sqrt;
    return sqrt(((outputs - targets)**2).sum()/len(outputs));


def accuracy(outputs, targets, threshold=0.5):
    """Simple function to calc the accuracy given a threshold"""
    from numpy import zeros, abs;
    zeroOnes = zeros(len(outputs));
    zeroOnes[outputs >= threshold] = 1.0;
    return (len(outputs) - abs(zeroOnes - targets).sum())/float(len(outputs));


def correlation(outputs, targets):
    """Simple function to calc the correlation."""
    from scipy import corrcoef;
    return corrcoef(outputs, targets)[0,1];
