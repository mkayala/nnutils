#!/usr/bin/env python
# encoding: utf-8
"""
Module created to rewrite a lot the of files in the above module to use a feature dictionary representation.

Idea is that data for the orb pair representation will be several files:  
- feature dicts representing an orb pair.
- indices of fdicts and 0/1 (maybe 0/.5/1) targets for predicted/not predicted. (For classification)
- pairs of indices of fdicts and 0/1 targets for greater/less (for pairwise classification) 

One method to normalize;

Then also, two types of trainers = 
- PairWiseTrainer takes fDict file and index pair file.  Trains shared weight nn
- ClassTrainer takes fDict file and a target file.., 

And two types of predictors.

__init__.py

Created by Matt Kayala on 2010-08-11.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.


"""
