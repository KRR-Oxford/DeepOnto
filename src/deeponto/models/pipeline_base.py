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
"""Class for running implemented models"""

from typing import Optional

from deeponto import SavedObj
from deeponto.onto import Ontology
from deeponto.utils import detect_path, create_path
from deeponto.config import InputConfig


class OntoPipelineBase:
    def __init__(
        self, model_name: str, saved_path: str, config_path: str,
    ):
        self.model_name = model_name
        self.paths = dict()
        self.paths["main_dir"] = saved_path
        self.config = InputConfig.load_config(config_path)
        
    def run(self):
        """Run the whole pipeline
        """
        create_path(self.paths["main_dir"])
        #TODO: do the pipeline

    def load_onto(self):
        """Load an input ontology if any
        """
        raise NotImplementedError

    def load_model(self):
        """Load alignment model according to model name
        """
        raise NotImplementedError

    @staticmethod
    def from_saved(saved_obj_path: str, is_onto: bool = False) -> Optional[SavedObj]:
        if detect_path(saved_obj_path):
            if not is_onto:
                return SavedObj.from_saved(saved_obj_path)
            else:
                return Ontology.from_saved(saved_obj_path)
        else:
            create_path(saved_obj_path)
            return None

    def complete_path(self, relative_path: str):
        main_dir = self.paths["main_dir"]
        return f"{main_dir}/{relative_path}"
