#!/usr/bin/env python
# encoding: utf-8
"""
Env.py

Created by Matt Kayala on 2010-08-11.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.
"""
"""Various constants for use by the application modules,
but these can / should be changed depending on the platform / environment
where they are installed.
"""

import sys
import logging

"""Default level for application logging.  Modify these for different scenarios.  
See Python logging package documentation for more information"""
#LOGGER_LEVEL = logging.DEBUG
LOGGER_LEVEL = logging.INFO
#LOGGER_LEVEL = logging.WARNING
#LOGGER_LEVEL = logging.ERROR
#LOGGER_LEVEL = logging.CRITICAL

#from CHEM.Common.Env import ORB_DB_PARAM;