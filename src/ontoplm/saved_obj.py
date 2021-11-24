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

import json


class SavedObj:
    @classmethod
    def from_new(cls, *args, **kwargs):
        """ constructor for new instance
        """
        raise NotImplementedError

    @classmethod
    def from_saved(cls, *args, **kwargs):
        """ constructor for loading saved instance
        """
        raise NotImplementedError
    
    def save_instance(self, *args, **kwargs):
        """ save the current instance locally
        """
        raise NotImplementedError

    @staticmethod
    def save_json(json_obj, saved_path: str, sort_keys: bool = False):
        with open(saved_path, "w") as f:
            json.dump(json_obj, f, indent=4, separators=(",", ": "), sort_keys=sort_keys)

    @staticmethod
    def load_json(saved_path: str):
        with open(saved_path, "r") as f:
            json_obj = json.load(f)
        return json_obj
