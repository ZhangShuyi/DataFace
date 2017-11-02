# -*- coding: utf-8 -*-
"""
    Version 1.1
    To implement the log function
    CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET
"""
from logging.handlers import RotatingFileHandler
import logging.config
import os

CURRENT_PATH = os.path.split(os.path.realpath(__file__))[0]
LOG_ROOT_FILE = os.path.join(CURRENT_PATH + '/../Log')
LOG_CONFIG_PATH = os.path.join(LOG_ROOT_FILE, "Log.config")

logging.config.fileConfig(LOG_CONFIG_PATH)

ROOT_LOG = logging.getLogger("root")
PRINT_LOG = logging.getLogger("print")
RECORD_LOG = logging.getLogger("record")
RECORD_LOG.propagate = False

RECORD_FILE = os.path.join(LOG_ROOT_FILE, "AllLog.log")
RecordHandler = RotatingFileHandler(RECORD_FILE, maxBytes=10 * 1024 * 1024, backupCount=10)
RecordHandler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(filename)s %(levelname)s  %(message)s")
RecordHandler.setFormatter(formatter)


ROOT_LOG.addHandler(RecordHandler)
RECORD_LOG.addHandler(RecordHandler)

ROOT_LOG.info("Start LOG_ROOT module (configuration {})".format(LOG_CONFIG_PATH))
PRINT_LOG.info("Start LOG_PRINT module")
RECORD_LOG.info("Start LOG_RECORD module")
