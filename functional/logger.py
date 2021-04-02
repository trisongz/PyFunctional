# Imports

import threading
import os
import sys
import logging
from typing import Optional

_lock = threading.Lock()
_loggerhandlers = {}


class LogFormatter(logging.Formatter):
    COLOR_CODES = {
        logging.CRITICAL: "\033[38;5;196m", # bright/bold magenta
        logging.ERROR:    "\033[38;5;9m", # bright/bold red
        logging.WARNING:  "\033[38;5;11m", # bright/bold yellow
        logging.INFO:     "\033[38;5;111m", # white / light gray
        logging.DEBUG:    "\033[1;30m"  # bright/bold black / dark gray
    }

    RESET_CODE = "\033[0m"
    def __init__(self, color, *args, **kwargs):
        super(LogFormatter, self).__init__(*args, **kwargs)
        self.color = color

    def format(self, record, *args, **kwargs):
        if (self.color == True and record.levelno in self.COLOR_CODES):
            record.color_on  = self.COLOR_CODES[record.levelno]
            record.color_off = self.RESET_CODE
        else:
            record.color_on  = ""
            record.color_off = ""
        return super(LogFormatter, self).format(record, *args, **kwargs)

class FunctionalLogger:
    def __init__(self, config):
        self.config = config
        self.logger = self.setup_logging()
    
    def setup_logging(self):
        if self.config['name'] != 'Functional':
            logger = logging.getLogger(name)
        else:
            logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logging.getLogger('gevent').setLevel(logging.ERROR)

        if self.config["console_log_output"] == "stdout":
            console_log_output = sys.stdout
        else:
            console_log_output = sys.stderr
        
        console_handler = logging.StreamHandler(console_log_output)
        console_handler.setLevel(self.config["console_log_level"].upper())
        console_formatter = LogFormatter(fmt=self.config["log_line_template"], color=self.config["console_log_color"])
        console_handler.setFormatter(console_formatter)
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.addHandler(console_handler)
        logger.propagate = False
        return logger
    
    def __call__(self, msg):
        if isinstance(msg, list):
            msg = '\n'.join(msg)
        elif isinstance(msg, dict):
            _msg, x = '', 0
            for k, v in msg.items():
                _msg += f' {k}: {v} |'
                x += 1
                if x % 3 == 0:
                    _msg += '\n'
            msg = _msg
        elif not isinstance(msg, str):
            msg = str(msg)
        self.logger.info(msg)

    
    def info(self, *args, **kwargs):
        return self.logger.info(*args, **kwargs)
    
    def warn(self, *args, **kwargs):
        return self.logger.warning(*args, **kwargs)
    
    def err(self, *args, **kwargs):
        return self.logger.error(*args, **kwargs)
    
    def d(self, *args, **kwargs):
        return self.logger.debug(*args, **kwargs)
    
    def log(self, *args, **kwargs):
        return self.info(*args, **kwargs)

    def get_logger(self):
        return self.logger
    

def _setup_library_root_logger(name):
    logger_config = {
        'name': name,
        'console_log_output': "stdout", 
        'console_log_level': "info",
        'console_log_color': True,
        'logfile_file': None,
        'logfile_log_level': "debug",
        'logfile_log_color': False,
        'log_line_template': f"%(color_on)s[{name}] %(funcName)-5s%(color_off)s: %(message)s"
    }
    logger = FunctionalLogger(logger_config)
    return logger


def _configure_library_root_logger(name="Functional") -> None:
    global _loggerhandlers
    with _lock:
        if name in _loggerhandlers:
            return
        _loggerhandlers[name] = _setup_library_root_logger(name)


def get_logger(name: Optional[str] = "Functional") -> logging.Logger:
    if name is None:
        name = "Functional"
    _configure_library_root_logger(name)
    return _loggerhandlers[name]

