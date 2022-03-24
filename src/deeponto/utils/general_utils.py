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
"""Providing useful utility functions"""
from typing import Optional
from pathlib import Path
import pandas as pd
import random
import re
import os
import torch
import numpy as np

##################################################################################
###                             element processing                             ###
##################################################################################


def uniqify(ls):
    """Return a list of unique elements without messing around the order
    """
    non_empty_ls = list(filter(lambda x: x != "", ls))
    return list(dict.fromkeys(non_empty_ls))


def to_identifier(var_name: str):
    """Change a variable name to a valid identifier
    """
    if var_name.isidentifier():
        return var_name
    else:
        changed_name = "".join(re.findall(r"[a-zA-Z_]+[0-9]*", var_name))
        print(f"change invalid identifier name: {var_name} ==> {changed_name}")
        return changed_name


def sort_dict_by_values(dic: dict, desc: bool = True, top_k: Optional[int] = None):
    """Return a sorted dict by values with top k reserved
    """
    top_k = len(dic) if not top_k else top_k
    sorted_items = list(sorted(dic.items(), key=lambda item: item[1], reverse=desc))
    return dict(sorted_items[:top_k])


def evenly_divide(start, end, num_splits: int):
    step = (end - start) / num_splits
    return [start + step * i for i in range(num_splits + 1)]

##################################################################################
###                                 randomness                                 ###
##################################################################################


def rand_sample_excl(start, end, number, *excl):
    """Randomly generate a number between {start} and {end} with end and specified 
    {excl} value(s) excluded
    """
    field = list(set(range(start, end)) - set(excl))
    if not field:
        raise ValueError(f"impossible to generate a number because the whole range is excluded")
    return random.sample(field, number)


##################################################################################
###                                     path                                   ###
##################################################################################


def detect_path(saved_obj_path: str) -> bool:
    """Check if path exists
    """
    return os.path.exists(saved_obj_path)


def create_path(path: str):
    """Create path recursively"""
    Path(path).mkdir(parents=True, exist_ok=True)


##################################################################################
###                                command line                                ###
##################################################################################


def print_choices(choices: list):
    """Print available choices
    """
    for i in range(len(choices)):
        print(f"[{i}]:", choices[i])


##################################################################################
###                              file processing                               ###
##################################################################################

# TODO: this might change with the version of pandas
na_vals = pd.io.parsers.readers.STR_NA_VALUES.difference({"NULL", "null", "n/a"})


def read_tsv(file_path: str):
    """Read tsv file as pandas dataframe without treating "null" as empty string.
    """
    return pd.read_csv(file_path, sep="\t", na_values=na_vals, keep_default_na=False)


##################################################################################
###                                 torch                                      ###
##################################################################################


def get_device(device_num: int = 0):
    """Get a device (GPU or CPU) for the torch model
    """
    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device(f"cuda:{device_num}")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(device_num))
    # If not...
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    return device


def set_seed(seed_val: int = 888):
    """Set random seed for reproducible results
    """
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

