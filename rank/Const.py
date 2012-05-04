#!/usr/bin/env python
# encoding: utf-8
"""
Const.py

Created by Matt Kayala on 2010-08-11.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.
"""
"""Various constants for use by the application modules"""
import Env;

"""Application name, for example to identify a common logger object"""
APPLICATION_NAME = "CHEM.ML.orbDB.featDict.learning.app"

"""Default level for application logging.  Modify these for different scenarios.  See Python logging package documentation for more information"""
LOGGER_LEVEL = Env.LOGGER_LEVEL

"""Default format of logger output"""
LOGGER_FORMAT = "[%(asctime)s %(levelname)s] %(message)s"

#from Env import ORB_DB_PARAM;

"""For ProgressDots"""
PROG_BIG = 10000;
PROG_SMALL = 500;
