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
import subprocess
import json
import ast
from datasets import load_dataset
from openprompt.data_utils import InputExample

##################################################################################
###                             element processing                             ###
##################################################################################


def uniqify(ls):
    """Return a list of unique elements without messing around the order
    """
    non_empty_ls = list(filter(lambda x: x != "", ls))
    return list(dict.fromkeys(non_empty_ls))


def parse_tuple(tuple_string: str):
    """Parse string tuple to tuple, e.g., '(1, 2)' => (1, 2)
    """
    return ast.literal_eval(tuple_string)


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


def split_java_identifier(java_style_identifier: str):
    """Split words in java's identifier style into natural language phrase.
    e.g.1, SuperNaturalPower => Super Natural Power.
    e.g.2, APIReference => API reference
    """
    raw_words = re.findall("([0-9A-Z][a-z]*)", java_style_identifier)
    # need to fix for the split of consecutive capitalized words such as API
    words = []
    capitalized_word = ""
    for i, w in enumerate(raw_words):
        # the above regex pattern will split at capitals
        # so the capitalized words are split into characters
        if len(w) == 1:
            capitalized_word += w
            # edge case for the last word
            if i == len(raw_words) - 1:
                words.append(capitalized_word)
        # if the the current w is a full word, save the previous cached capitalized_word
        elif capitalized_word:
            words.append(capitalized_word)
            words.append(w.lower())
            capitalized_word = ""
        else:
            words.append(w.lower())

    return words


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


def read_table(file_path: str):
    """Read tsv file as pandas dataframe without treating "null" as empty string.
    """
    sep = "\t" if file_path.endswith(".tsv") else ","
    return pd.read_csv(file_path, sep=sep, na_values=na_vals, keep_default_na=False)


def read_jsonl(file_path: str):
    """Read .jsonl file (list of json) introduced in the BLINK project.
    """
    results = []
    key_set = []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
        for line in lines:
            record = json.loads(line)
            results.append(record)
            key_set += list(record.keys())
    print(f"all available keys: {set(key_set)}")
    return results


def data_split(tsv_mapping_path: str, out_dir: str):
    """Split mapping data into unsupervised val:test = 1:9
    and semi-supervised settings train:val:test = 2:1:7.
    """
    map_df = read_table(tsv_mapping_path)
    train_and_val = map_df.sample(frac=0.3)
    val = train_and_val.sample(frac=1 / 3)
    train = train_and_val.drop(index=val.index)
    test = map_df.drop(index=train_and_val.index)
    assert all(train.append(val).append(test).sort_index() == map_df)

    create_path(out_dir + "/semi_supervised")
    create_path(out_dir + "/unsupervised")
    train.to_csv(out_dir + "/semi_supervised/train.tsv", sep="\t", index=False)
    val.to_csv(out_dir + "/semi_supervised/val.tsv", sep="\t", index=False)
    val.to_csv(out_dir + "/unsupervised/val.tsv", sep="\t", index=False)
    test.to_csv(out_dir + "/semi_supervised/test.tsv", sep="\t", index=False)
    train.append(test).to_csv(out_dir + "/unsupervised/test.tsv", sep="\t", index=False)
    train.append(val).to_csv(out_dir + "/semi_supervised/train+val.tsv", sep="\t", index=False)


def load_prompt_data(dataset_path: str = "multi_nli", *splits: str):
    """Load a datast from huggingface datasets and transform to openprompt examples.
    """
    data_dict = load_dataset(dataset_path)
    prompt_data_dict = dict()
    splits = list(splits) if splits else data_dict.keys()
    i = 0
    for split in splits:
        x_samples = []
        for samp in data_dict[split]:
            inp = InputExample(guid=i, meta=samp, label=samp["label"])
            x_samples.append(inp)
            i += 1
        prompt_data_dict[split] = x_samples
    return prompt_data_dict


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


##################################################################################
###                                 java                                       ###
##################################################################################


def run_jar(command: str):
    """Run jar command using subprocess
    """
    proc = subprocess.Popen(command.split(" "))
    try:
        _, _ = proc.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, _ = proc.communicate()
