nnutils: Neural Network Training for Sparse Data
================================================

Implementation of neural network techniques for classification
and ranking with sparse data in Python.  This is based off of 
the  [Monte gradient based learning](http://montepython.sourceforge.net/) 
package by Roland Memisevic.

Includes:

* Sparse feature dictionary file format
* Per-weight adaptive momentum stochastic gradient descent
* Several other gradient descent trainers
* Implementation of simple classification
* Implementation of the 
  [RankNet](http://research.microsoft.com/en-us/um/people/cburges/papers/ICML_ranking.pdf) 
  pairwise ranking model


* By Matt Kayala, University of California, Irvine.  Main code 
  licensed under GPL, included monte version licensed under 
  included license file.  See LICENSE.txt and monte/LICENSE 
  for details.


Details of included Monte
-------------------------

The included monte module is based on verion 0.1.0 of the 
[main code](http://montepython.sourceforge.net/).  Minor 
changes have been made:

* All pylab imports have been removed (as they are unnecessary).
* Auto importing of sub-modules has been removed. 

Requirements
------------

The code requires Python 2.5+ (though the tests might only work in 2.7),
NumPy 1.6+, and SciPy 0.9.0+. 

Simply place on your python path and you can 

  import nnutils

### Acknowledements

The sparse feature file representation is based off of code from 
Jonathan Chen and Josh Swamidass at the University of California,
Irvine.