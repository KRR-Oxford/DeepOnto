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
"""Utility functions for file processing."""

from __future__ import annotations

import json
import yaml
import dill as pickle
import os
import shutil
from pathlib import Path
import pandas as pd
from lxml import etree, builder


class FileProcessor:
    def __init__(self):
        """Provides file processing utilities.
        """
        pass
    
    @staticmethod
    def detect_path(path: str) -> bool:
        """Check if a path exists.
        """
        return os.path.exists(path)

    @staticmethod
    def create_path(path: str):
        """Create a path recursively.
        """
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_file(obj, save_path: str, sort_keys: bool = False):
        """Save an object to a certain format.
        """
        if save_path.endswith(".json"):
            with open(save_path, "w") as output:
                json.dump(obj, output, indent=4, separators=(",", ": "), sort_keys=sort_keys)
        elif save_path.endswith(".pkl"):
            with open(save_path, "wb") as output:
                pickle.dump(obj, output, -1)
        elif save_path.endswith(".yaml"):
            with open(save_path, "w") as output:
                yaml.dump(obj, output, default_flow_style=False, allow_unicode=True)
        else:
            raise RuntimeError(f"Unsupported saving format: {save_path}")
    
    @staticmethod
    def load_file(save_path: str):
        """Load an object of a certain format.
        """
        if save_path.endswith(".json"):
            with open(save_path, "r") as input:
                return json.load(input)
        elif save_path.endswith(".pkl"):
            with open(save_path, "rb") as input:
                return pickle.load(input)
        elif save_path.endswith(".yaml"):
            with open(save_path, "r") as input:
                return yaml.safe_load(input)
        else:
            raise RuntimeError(f"Unsupported loading format: {save_path}")

    @staticmethod
    def print_dict(dic: dict):
        """Pretty print a dictionary.
        """
        pretty_print = json.dumps(dic, indent=4, separators=(",", ": "))
        # print(pretty_print)
        return pretty_print

    @staticmethod
    def copy2(source: str, destination: str):
        """Copy a file from source to destination.
        """
        try:
            shutil.copy2(source, destination)
            print(f"copied successfully FROM {source} TO {destination}")
        except shutil.SameFileError:
            print(f"same file exists at {destination}")

    # @staticmethod
    # def report(root_name: str, **kwargs) -> str:
    #     """Generate xml report for the saved object
    #     """
    #     xml = builder.ElementMaker()
    #     # root_name = type(self).__name__ if not root_name else root_name
    #     elems = []
    #     for k, v in kwargs.items():
    #         elems.append(getattr(xml, k)(str(v)))
    #     root = getattr(xml, root_name)(*elems)
    #     string = etree.tostring(root, pretty_print=True).decode()
    #     return string


    def read_table(table_file_path: str):
        r"""Read `csv` or `tsv` file as pandas dataframe without treating `"NULL"`, `"null"`, and `"n/a"` as an empty string.
        """
        # TODO: this might change with the version of pandas
        na_vals = pd.io.parsers.readers.STR_NA_VALUES.difference({"NULL", "null", "n/a"})
        sep = "\t" if table_file_path.endswith(".tsv") else ","
        return pd.read_csv(table_file_path, sep=sep, na_values=na_vals, keep_default_na=False)


    def read_jsonl(file_path: str):
        """Read `.jsonl` file (list of json) introduced in the BLINK project.
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
