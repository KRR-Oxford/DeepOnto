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

from collections import defaultdict
from typing import Optional, List, Tuple, TYPE_CHECKING
import pandas as pd
import ast
import pprintpp

from deeponto.onto import Ontology
from deeponto.utils import FileUtils

if TYPE_CHECKING:
    from org.semanticweb.owlapi.model import OWLObject  # type: ignore

DEFAULT_REL = "<?rel>"
DUP_STRATEGIES = ["average", "kept_new", "kept_old"]
DEFAULT_DUP_STRATEGY = DUP_STRATEGIES[0]


##################################################################################
###                         basic mapping structure                            ###
##################################################################################


class EntityMapping:
    r"""A datastructure for entity mapping.
    
    Such entities should be named and have an IRI.

    Attributes:
        src_entity_iri (str): The IRI of the source entity, usually its IRI if available.
        tgt_entity_iri (str): The IRI of the target entity, usually its IRI if available.
        relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
            Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.
        score (float, optional): The score that indicates the confidence of this mapping. Defaults to `0.0`.
    """
    
    def __init__(
        self, src_entity_iri: str, tgt_entity_iri: str, relation: str = DEFAULT_REL, score: float = 0.0
    ):
        """Intialise an entity mapping.

        Args:
            src_entity_iri (str): The IRI of the source entity, usually its IRI if available.
            tgt_entity_iri (str): The IRI of the target entity, usually its IRI if available.
            relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
                Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.
            score (float, optional): The score that indicates the confidence of this mapping. Defaults to `0.0`.
        """
        self.head = src_entity_iri
        self.tail = tgt_entity_iri
        self.relation = relation
        self.score = score
        
    @classmethod
    def from_owl_objects(cls, src_entity: OWLObject, tgt_entity: OWLObject, relation: str = DEFAULT_REL, score: float = 0.0):
        """Create an entity mapping from two `OWLObject` entities which have an IRI.

        Args:
            src_entity (OWLObject): The source entity in `OWLObject`.
            tgt_entity (OWLObject): The target entity in `OWLObject`.
            relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
                Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.
            score (float, optional): The score that indicates the confidence of this mapping. Defaults to `0.0`.
            
        Returns:
            (EntityMapping): The entity mapping created from the source and target entities.
        """
        return cls(str(src_entity.getIRI()), str(tgt_entity.getIRI()), relation, score)

    def to_tuple(self):
        """Get the tuple of (head, tail) and discard the relation and score.
        """
        return (self.head, self.tail)
    
    @staticmethod
    def sort_entity_mappings_by_score(entity_mappings: List[EntityMapping], k: Optional[int] = None):
        r"""Sort the entity mappings in a list by their scores in descending order.
        
        Args:
            entity_mappings (List[EntityMapping]): A list entity mappings to sort.
            k (int, optional): The number of top $k$ scored entities preserved if specified. Defaults to None which 
            means to return *all* entity mappings.
            
        Returns:
            (List[EntityMapping]): A list of sorted entity mappings.
        """
        return list(sorted(entity_mappings, key=lambda x: x.score, reverse=True))[:k]
    
    @staticmethod
    def read_table_mappings(
        table_of_mappings_file: str,
        threshold: Optional[float] = 0.0,
        relation: str = DEFAULT_REL,
        is_reference: bool = False
    ):
        r"""Read entity mappings from `.csv` or `.tsv` files.

        Args:
            table_of_mappings_file (str): The path to the table (`.csv` or `.tsv`) of mappings.
            threshold (Optional[float], optional): Mappings with scores less than `threshold` will not be loaded. Defaults to 0.0.
            relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
                Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.
            is_reference (bool): Whether the loaded mappings are reference mappigns; if so, `threshold` is disabled and mapping scores 
            are all set to $1.0$. Defaults to `False`.

        Returns:
            List[EntityMapping]: A list of entity mappings loaded from the table file.
        """
        df = FileUtils.read_table(table_of_mappings_file)
        entity_mappings = []
        for _, dp in df.iterrows():
            if is_reference:
                entity_mappings.append(ReferenceMapping(dp["SrcEntity"], dp["TgtEntity"], relation))
            else:
                if dp["Score"] >= threshold:
                    entity_mappings.append(EntityMapping(dp["SrcEntity"], dp["TgtEntity"], relation, dp["Score"]))
        return entity_mappings

    def __repr__(self):
        return f"EntityMapping({self.head} {self.relation} {self.tail}, {round(self.score, 6)})"


class ReferenceMapping(EntityMapping):
    r"""A datastructure for entity mapping that acts as a reference mapping.
    
    A reference mapppings is a ground truth entity mapping (with $score = 1.0$) and can
    have several entity mappings as candidates. These candidate mappings should have the 
    same `head` (i.e., source entity) as the reference mapping.

    Attributes:
        src_entity_iri (str): The IRI of the source entity, usually its IRI if available.
        tgt_entity_iri (str): The IRI of the target entity, usually its IRI if available.
        relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
            Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.
    """

    def __init__(
        self,
        src_ent_iri: str,
        tgt_ent_iri: str,
        relation: str = DEFAULT_REL,
        candidate_mappings: Optional[List[EntityMapping]] = []
    ):
        r"""Intialise a reference mapping.

        Args:
            src_entity_iri (str): The IRI of the source entity, usually its IRI if available.
            tgt_entity_iri (str): The IRI of the target entity, usually its IRI if available.
            relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
                Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.
            candidate_mappings (List[EntityMapping], optional): A list of entity mappings that are candidates for this reference mapping. Defaults to `[]`.
        """
        super().__init__(src_ent_iri, tgt_ent_iri, relation, 1.0)
        self.candidates = []
        for candidate in candidate_mappings:
            self.add_candidate(candidate)

    def __repr__(self):
        reference_mapping_str = f"ReferenceMapping({self.head} {self.relation} {self.tail}, 1.0)"
        if self.candidates:
            candidate_mapping_str = pprintpp.pformat(self.candidates)
            reference_mapping_str += f" with candidates:\n{candidate_mapping_str}"
        return reference_mapping_str

    def add_candidate(self, candidate_mapping: EntityMapping):
        """Add a candidate mapping whose relation and head entity are the
        same as the reference mapping's.
        """
        if self.relation != candidate_mapping.relation:
            raise ValueError(
                "Expect relation of candidate mapping to " + f"be {self.relation} but got {candidate_mapping.relation}"
            )
        if self.head != candidate_mapping.head:
            raise ValueError(
                "Candidate mapping does not have the same head entity as the anchor mapping."
            )
        self.candidates.append(candidate_mapping)
        
    @staticmethod
    def read_table_mappings(
        table_of_mappings_file: str,
        relation: str = DEFAULT_REL
    ):
        r"""Read reference mappings from `.csv` or `.tsv` files.

        Args:
            table_of_mappings_file (str): The path to the table (`.csv` or `.tsv`) of mappings.
            relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
                Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.

        Returns:
            List[ReferenceMapping]: A list of reference mappings loaded from the table file.
        """
        return EntityMapping.read_table_mappings(table_of_mappings_file, relation=relation, is_reference=True)
