#!/usr/bin/env python
# encoding: utf-8
"""
Const.py

Created by Matt Kayala on 2010-05-04.
"""
"""Various constants for use by the application modules"""
import Env;

"""Application name, for example to identify a common logger object"""
APPLICATION_NAME = "nnutils.mutil"

"""Default level for application logging.  Modify these for different scenarios.  See Python logging package documentation for more information"""
LOGGER_LEVEL = Env.LOGGER_LEVEL

"""Default format of logger output"""
LOGGER_FORMAT = "[%(asctime)s %(levelname)s] %(message)s"

## If we get numbers that are close to zero or close to one, offset by this much.
OFFSET_EPSILON = 1e-8;

EPSILON = 1e-12;
