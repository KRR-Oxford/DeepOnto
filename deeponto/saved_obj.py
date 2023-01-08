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
"""Super class for objects that can be created from new or from saved"""

from __future__ import annotations

import json
import dill as pickle
import os
import shutil
from typing import Optional
from pathlib import Path
from lxml import etree, builder


class SavedObj:
    def __init__(self, saved_name):
        self.saved_name = saved_name

    @classmethod
    def from_new(cls, *args, **kwargs):
        """constructor for new instance
        """
        raise NotImplementedError

    @classmethod
    def from_saved(cls, saved_path, *args, **kwargs):
        """constructor for loading saved instance
        """
        return cls.load_pkl(saved_path)

    def save_instance(self, saved_path, *args, **kwargs):
        """save the current instance locally
        """
        Path(saved_path).mkdir(parents=True, exist_ok=True)
        self.save_pkl(self, saved_path)

    @staticmethod
    def save_json(json_obj, saved_path: str, sort_keys: bool = False):
        with open(saved_path, "w") as f:
            json.dump(json_obj, f, indent=4, separators=(",", ": "), sort_keys=sort_keys)

    @staticmethod
    def load_json(saved_path: str) -> dict:
        with open(saved_path, "r") as f:
            json_obj = json.load(f)
        return json_obj

    @staticmethod
    def print_json(json_obj):
        print(json.dumps(json_obj, indent=4, separators=(",", ": ")))

    @staticmethod
    def save_pkl(obj: SavedObj, saved_path: str):
        saved_path = saved_path + f"/{obj.saved_name}.pkl"
        with open(saved_path, "wb") as output:
            pickle.dump(obj, output, -1)

    @staticmethod
    def load_pkl(saved_path: str):
        """load the pickled part of the SavedObj
        """
        for file in os.listdir(saved_path):
            if file.endswith(".pkl"):
                with open(f"{saved_path}/{file}", "rb") as input:
                    obj = pickle.load(input)
                return obj

    @staticmethod
    def copy2(source, destination):
        try:
            shutil.copy2(source, destination)
            print(f"copied successfully FROM {source} TO {destination}")
        except shutil.SameFileError:
            print(f"same file exists at {destination}")

    def report(self, root_name: Optional[str] = None, **kwargs) -> str:
        """generate xml report for the saved object
        """
        xml = builder.ElementMaker()
        root_name = type(self).__name__ if not root_name else root_name
        elems = []
        for k, v in kwargs.items():
            elems.append(getattr(xml, k)(str(v)))
        root = getattr(xml, root_name)(*elems)
        string = etree.tostring(root, pretty_print=True).decode()
        return string
