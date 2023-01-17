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
"""Utility functions for general purposes."""

from __future__ import annotations

from functools import wraps
import time


def timer(function):
    """Print the runtime of the decorated function."""

    @wraps(function)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = function(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {function.__name__!r} in {run_time:.4f} secs.")
        return value

    return wrapper_timer


def debug(function):
    """Print the function signature and return value."""

    @wraps(function)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {function.__name__}({signature})")
        value = function(*args, **kwargs)
        print(f"{function.__name__!r} returned {value!r}.")
        return value

    return wrapper_debug


def paper(title: str, link: str):
    """Add paper tagger for methods."""
    # Define a new decorator, named "decorator", to return
    def decorator(func):
        # Ensure the decorated function keeps its metadata
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call the function being decorated and return the result
            return func(*args, **kwargs)

        wrapper.paper_title = f'This method is associated with tha paper of title: "{title}".'
        wrapper.paper_link = f"This method is associated with the paper with link: {link}."
        return wrapper

    # Return the new decorator
    return decorator
