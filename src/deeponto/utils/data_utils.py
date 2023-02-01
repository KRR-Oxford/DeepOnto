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

from __future__ import annotations

from typing import Optional

class DataUtils:
    
    @staticmethod
    def uniqify(ls):
        """Return a list of unique elements without messing around the order"""
        non_empty_ls = list(filter(lambda x: x != "", ls))
        return list(dict.fromkeys(non_empty_ls))

    @staticmethod
    def sort_dict_by_values(dic: dict, desc: bool = True, k: Optional[int] = None):
        """Return a sorted dict by values with first k reserved if provided."""
        sorted_items = list(sorted(dic.items(), key=lambda item: item[1], reverse=desc))
        return dict(sorted_items[:k])
