import Const
import sys, os
import logging
import cgi, UserDict
import unittest

log = logging.getLogger(Const.APPLICATION_NAME)
log.setLevel(Const.LOGGER_LEVEL)

handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter(Const.LOGGER_FORMAT)

handler.setFormatter(formatter)
log.addHandler(handler)

### A function for testing feature dicts:
from nnutils.test.Util import spectrumExtractor
