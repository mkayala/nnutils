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
