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
"""Base Class for Ontology Subsumption Inference Data Sampling"""

import itertools
import random
from typing import Callable, Optional, List
import enlighten
from collections import defaultdict

from deeponto.onto import Ontology
from deeponto.onto.logic.reasoner import OWLReasoner
from deeponto import SavedObj, OWL_THING


class SubsumptionSamplerBase:
    def __init__(
        self,
        onto: Ontology,
        neg_ratio: int = 1
    ):
        self.onto = onto
        self.reasoner = OWLReasoner(onto.owl_path)
        self.neg_ratio = neg_ratio
        self.progress_manager = enlighten.get_manager()
        self.subs = {}
        
    def init_subs(self):
        raise NotImplementedError
    
    def save(self, saved_path: str):
        """Save in json format similar to the huggingface datasets manner
        """
        SavedObj.save_json(self.subs, saved_path)
        
    def random_atomic_class(self):
        """Randomly draw a named concept's IRI
        """
        return random.choice(self.reasoner.class_iris)
    
    def random_sibling_atomic_class_for(self, class_iri: str):
        """Randomly draw a sibling class for a given class
        """
        try:
            return random.choice(self.reasoner.sibling_dict[class_iri])
        except:
            return None
    
    def random_atomic_class_pair(self):
        """Randomly draw a pair of named concepts' IRIs
        """
        return tuple(random.sample(self.reasoner.class_iris, k=2))
        
    def random_sibling_atomic_class_pair(self):
        """Randomly draw a pair of named concepts' IRIs that are sibling
        """
        return tuple(random.choice(self.reasoner.sibling_pairs))
