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
"""Generate Subsumption Mappings from Equivalence Mappings"""


from deeponto.onto.text.text_utils import unfold_iri
from deeponto.onto import Ontology
from deeponto.onto.mapping import OntoMappings
from deeponto import FlaggedObj


class SubsumptionMappingGenerator(FlaggedObj):
    def __init__(
        self,
        src_onto: Ontology,
        tgt_onto: Ontology,
        equiv_mappings_path: str,
        map_ratio: int = 1,
    ):
        super().__init__(src_onto, tgt_onto)
        self.equiv_pairs = OntoMappings.read_tsv_mappings(equiv_mappings_path).to_tuples()
        self.map_ratio = map_ratio
        
    def generate_from_equiv_pair(self, src_ent_name: str, tgt_ent_name: str):
        
        pass
        src_iri = unfold_iri(src_ent_name)
        

    def equiv_src_iris(self):
        """Return iris of the source classes from the equivalence mapping
        for downstream class deletion
        """
        return [unfold_iri(p[0]) for p in self.equiv_pairs]

    def equiv_tgt_iris(self):
        """Return iris of the target classes from the equivalence mapping
        for downstream class deletion
        """
        return [unfold_iri(p[1]) for p in self.equiv_pairs]
