# Copyright 2021 Yuan He. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional
import json
from transformers import set_seed as t_set_seed


def set_seed(seed):
    """Set seed function imported from transformers."""
    t_set_seed(seed)


def sort_dict_by_values(dic: dict, desc: bool = True, k: Optional[int] = None):
    """Return a sorted dict by values with first k reserved if provided."""
    sorted_items = list(sorted(dic.items(), key=lambda item: item[1], reverse=desc))
    return dict(sorted_items[:k])


def uniqify(ls):
    """Return a list of unique elements without messing around the order"""
    non_empty_ls = list(filter(lambda x: x != "", ls))
    return list(dict.fromkeys(non_empty_ls))


def print_dict(dic: dict):
    """Pretty print a dictionary."""
    pretty_print = json.dumps(dic, indent=4, separators=(",", ": "))
    # print(pretty_print)
    return pretty_print
