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

from typing import List, Optional
from yacs.config import CfgNode
import os

from deeponto.align.mapping import ReferenceMapping
from deeponto.onto import Ontology
from deeponto.utils import FileUtils



MODEL_OPTIONS = {
    "bertmap": {"trainable": True}, 
    "bertmaplt": {"trainable": False}
}

DEFAULT_CONFIG_FILE = os.join(os.path.dirname(__file__), "default_config.yaml")

def load_config(config_file: Optional[str] = None):
    """Load the configuration in `.yaml`. If the file
    is not provided, use the default configuration.
    """
    if not config_file:
        config_file = DEFAULT_CONFIG_FILE
        print(f"Use the default configuration at {DEFAULT_CONFIG_FILE}.")
    if not config_file.endswith(".yaml"):
        raise RuntimeError("Configuration file should be in `yaml` format.")
    return CfgNode(FileUtils.load_file(config_file))


class BERTMap:
    """Configurations for BERTMap.
    
    
    """
    
    def __init__(self, config_file: Optional[str] = None):
        
        self.config = load_config(config_file)
        self.name = self.config.matching.model
        self.output_path = os.path.abspath(self.config.output_path)
        assert self.name in MODEL_OPTIONS.keys()
        self.trainable = MODEL_OPTIONS[self.name]["trainable"]
        
        # ontology
        self.src_onto = Ontology(self.config.ontology.src_onto)
        self.tgt_onto = Ontology(self.config.ontology.tgt_onto)
        self.annotation_property_iris = self.config.ontology.annotation_property_iris
        
        # model
        self.bert_config = self.config.matching.bert
        
        # provided mappings if any
        self.training_mappings = self.config.matching.training_mappings
        if self.training_mappings:
            self.training_mappings = ReferenceMapping.read_table_mappings(self.training_mappings)
        self.validation_mappings = self.config.matching.validation_mappings
        if self.validation_mappings:
            self.validation_mappings = ReferenceMapping.read_table_mappings(self.validation_mappings)
        
                
