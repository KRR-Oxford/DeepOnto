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
"""Class for Ontology Prompting based on OpenPrompt"""

from __future__ import annotations

from openprompt.data_utils import InputExample
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification, PromptForGeneration
from openprompt import PromptDataLoader
from openprompt.plms import load_plm

from typing import TYPE_CHECKING, Union, List, Optional, Callable, Dict

if TYPE_CHECKING:
    from deeponto.onto import Ontology
    from owlready2.entity import ThingClass
    from owlready2.class_construct import Restriction

from . import OntoTemplate

class OntoPrompt:
    
    def __init__(self, onto: Ontology, plm_path: str, plm_type: str = "bert"):
        self.onto = onto
        self.plm_path = plm_path
        self.plm_type = plm_type
        
        # template for text processing
        self.onto_template = OntoTemplate(onto)

        # plm
        self.plm, self.tokenizer, self.model_config, self.wrapper_class = load_plm(
            plm_type, plm_path
        )

        # openprompt
        self.prompt_classes = ["true", "false"]
        self.prompt_template = ManualTemplate(
            text = '{"placeholder":"text_a"} . true or false ? {"mask"} .',
            tokenizer = self.tokenizer,
        )
        self.prompt_verbalizer =  ManualVerbalizer(
            classes = self.prompt_classes,
            label_words = {
                "true": ["true", "yes", "correct"],
                "false": ["false", "no", "incorrect"],
            },
            tokenizer = self.tokenizer,
        )
        self.prompt_model = PromptForClassification(
            template = self.prompt_template,
            plm = self.plm,
            verbalizer = self.prompt_verbalizer,
        )
        