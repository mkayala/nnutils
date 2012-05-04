#!/usr/bin/env python
# encoding: utf-8
"""
Util.py

Created by Matt Kayala on 2010-08-11.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.
"""
"""Miscellaneous utility functions used across the application
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

from CHEM.Common.Util import ProgressDots


