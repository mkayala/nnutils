#!/usr/bin/env python
# encoding: utf-8
"""
Some basic constants

File: Const.py
Author: MKayala
Created 2012-05-04
Copyright 2012. 
"""
import logging

"""Delimiter for feature and matrix files"""
TEXT_DELIM = "\t"

"""Delimiter for atom pair weight specification strings and feature index:count mappings"""
KEY_DELIM = ":"
ITEM_DELIM = ","

"""Prefix to identify feature:index mapping rows of feature dictionary text files"""
FEATURE_PREFIX = "#"

"""Define a really small number to check convergence and to avoid division by zero."""
EPSILON = 1E-12;

"""Updates to process before reporting progress"""
PROG_BIG = 1000;
PROG_SMALL = 25;

import Env;

"""Application name, for example to identify a common logger object"""
APPLICATION_NAME = "nnutils.classify"

"""Default level for application logging.  Modify these for different scenarios.
See Python logging package documentation for more information"""
LOGGER_LEVEL = Env.LOGGER_LEVEL

"""Default format of logger output"""
LOGGER_FORMAT = "[%(asctime)s %(levelname)s] %(message)s"
