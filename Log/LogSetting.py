# -*- coding: utf-8 -*-
"""
    Version 1.0
    To implement the log function
    CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET
"""
import logging
import logging.config

logging.config.fileConfig("Log.config")
LOGGER_ROOT = logging.getLogger("root")
LOGGER_PRINT = logging.getLogger("print")
LOGGER_RECORD = logging.getLogger("record")

LOGGER_ROOT.info("Start LOG_ROOT module")
LOGGER_PRINT.info("Start LOG_PRINT module")
LOGGER_RECORD.info("Start LOG_RECORD module")
