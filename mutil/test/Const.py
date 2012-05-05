"""Various constants for use by the mutil module test cases"""
import logging

"""Application name, for example to identify a common logger object"""
APPLICATION_NAME = "nnutils.mutil.test"

"""Default level for application logging.  Modify these for different scenarios.  
See Python logging package documentation for more information
"""
LOGGER_LEVEL = logging.INFO
LOGGER_LEVEL = logging.DEBUG

"""Default format of logger output"""
LOGGER_FORMAT = "[%(asctime)s %(levelname)s] %(message)s"

"""Verbosity of the test runner"""
RUNNER_VERBOSITY = 2

"""Application logger level.  Set this to higher level to suppress uninteresting
application output during test runs."""
APP_LOGGER_LEVEL = logging.CRITICAL
APP_LOGGER_LEVEL = logging.DEBUG

