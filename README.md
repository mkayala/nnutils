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


Details of included Monte:
--------------------------

The included monte module is based on verion 0.1.0 of the 
[main code](http://montepython.sourceforge.net/).  Minor 
changes have been made:

* All pylab imports have been removed (as they are unnecessary).
* Auto importing of sub-modules has been removed. 

Requirements:
-------------

The code requires Python 2.5+ (though the tests might only work in 2.7),
NumPy 1.6+, and SciPy 0.9.0+. 

Simply place on your python path and you can 

```python
import nnutils
```

How to Use:
-----------
Below are some short examples about how to use for classification.
The rank usage is similar, but requires a slightly different id file 
format.  Run the python scripts with --help to see available options.

### Write a dictionary to file in sparse format:

Given a list of dictionaries, write them to a zipped file.
```python  
import gzip
from nnutils.Util import FeatureDictWriter

fdlist = [{'a':1, 'b':2}, {'b':3, 'z':4}]
ofs = gzip.open('raw.fd.gz', 'w')
writer = FeatureDictWriter(ofs)
[writer.update(d) for d in fdlist]
ofs.close()
```

### Normalize a dictionary

Given a zipped feature dict and a mapfile (pickled dictionary mapping keys to positions in 
a feature vector), calculate normalization params, and normalize data.  (Right now only 
supports shift/scale to [-1, 1] range).  Can also normalize another file given the params 
calculated from a first fd file.

```bash
python nnutils/NormalizeFeatDicts.py -p normparams.pkl mapfile.pkl raw.fd.gz norm.fd.gz

python nnutils/NormFeatDictFromParams.py mapfile.pkl normparams.pkl otherraw.fd.gz othernorm.fd.gz
```

### Setup model architecture

Given a mapfile (pickled dictionary mapping fd keys to positions in feature vector), and command
line options, write out a pickled file containing the architecture settings for a nn model. 

```bash
python nnutils/classify/MakeArchFileModel.py --numhidden=10 --decay=0.1 --online mapfile.pkl archmodel.pkl
```

Run with --help to see all available options.

### Train a classifier 

Given the normalized fd file, a setup arch model file, and a white-space delimited id file of the 
format [rowid, otherid, class[0/1]], train a model:

```bash
python nnutils/classify/FDictClassTrainer.py archmodel.pkl norm.fd.gz idfile.txt train.archmodel.pkl
```

### Predict using the trained model

```bash
python nnutils/classify/FDictClassPredictor.py train.archmodel.pkl norm.fd.gz idfile.txt predictions.txt
```

Acknowledements
---------------

The sparse feature file representation is based off of code from 
Jonathan Chen and Josh Swamidass at the University of California,
Irvine.