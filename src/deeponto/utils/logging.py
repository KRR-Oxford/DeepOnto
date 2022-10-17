# Copyright 2021 Yuan He (KRR-Oxford). All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import datetime
import time
from functools import wraps

# subclass of logging.Formatter
class RuntimeFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()

    def formatTime(self, record, datefmt=None):
        """Record relative runtime in hr:min:sec format
        """
        duration = datetime.datetime.utcfromtimestamp(record.created - self.start_time)
        elapsed = duration.strftime("%H:%M:%S")
        return "{}".format(elapsed)


def create_logger(model_name: str, saved_path: str):
    """Create logger for both console info and saved info
    """
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(f"{saved_path}/{model_name}.log")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = RuntimeFormatter(
        "[Time: %(asctime)s] - [PID: %(process)d] - [Model: %(name)s] \n%(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


##################################################################################
###                              pretty print                                  ###
##################################################################################


def banner_msg(message: str, banner_len: int = 70, sym="#"):
    """Print banner message:
    
    ######################################################################
    ###                            example                             ###
    ######################################################################
    
    """
    print()
    print(sym * banner_len)
    message = sym * 3 + " " * ((banner_len - len(message)) // 2 - 3) + message
    message = message + " " * (banner_len - len(message) - 3) + sym * 3
    print(message)
    print(sym * banner_len)
    print()


##################################################################################
###                              adding credit                                 ###
##################################################################################

# NOTE: https://stackoverflow.com/questions/308999/what-does-functools-wraps-do
def credit(name: str, link: str):
    """Add credit tagger to any external functions"""
    # Define a new decorator, named "decorator", to return
    def decorator(func):
        # Ensure the decorated function keeps its metadata
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call the function being decorated and return the result
            return func(*args, **kwargs)
        wrapper.credit_to = f"This function is credited to {name} at {link}." 
        return wrapper
    # Return the new decorator
    return decorator
