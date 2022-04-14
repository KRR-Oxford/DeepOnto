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
"""Class for handling the ontology in owlready2 format

Known issues for owlready2:
- No new object will be created when loading ontologies of the same IRIs 
    - Solution: beforing handling the second owl of same IRI, destroy the cache in the first one

"""

from __future__ import annotations

import os
from typing import Optional, List
from collections import defaultdict
from owlready2 import get_ontology
from pyats.datastructures import AttrDict
from pathlib import Path

from deeponto import SavedObj
from .text import Tokenizer, text_utils


class Ontology(SavedObj):
    def __init__(self, owl_path: str):
        self.owl_path = os.path.abspath(owl_path)
        self.owl = get_ontology(f"file://{owl_path}").load()
        self.lab_props = None
        # dict attributes
        self.class2idx = None
        self.idx2class = None
        self.idx2labs = None
        self.inv_idx = None
        # stat attributes
        self.num_classes = None
        self.num_labs = None
        self.avg_labs = None
        self.num_entries_inv_idx = None
        super().__init__(self.owl.name)

    @classmethod
    def from_new(
        cls, onto_path: str, lab_props: List[str] = ["label"], tokenizer: Optional[Tokenizer] = None
    ) -> Ontology:
        onto = cls(onto_path)
        # {class_iri: class_number}; {class_number: class_iri}
        onto.class2idx, onto.idx2class = onto.assign_class_numbers(onto.owl)
        # {class_number: [labels]}
        onto.lab_props = lab_props
        onto.idx2labs, onto.num_labs = text_utils.ents_labs_from_props(
            onto.owl.classes(), onto.class2idx, onto.lab_props
        )
        onto.num_classes = len(onto.class2idx)
        onto.avg_labs = round(onto.num_labs / onto.num_classes, 2)
        # {token: [class_numbers]}
        if tokenizer:
            onto.build_inv_idx(tokenizer, cut=0)
        print(onto)
        return onto

    @classmethod
    def from_saved(cls, saved_path: str) -> Optional[Ontology]:
        try:
            onto = cls.load_pkl(saved_path)
            owl_file_name = onto.owl_path.split("/")[-1]
            onto.owl_path = saved_path + "/" + owl_file_name
            onto.owl = get_ontology(f"file://{onto.owl_path}").load()
            return onto
        except:
            raise FileNotFoundError(f"please check file integrity in : {saved_path})")

    def save_instance(self, saved_path: str):
        """The saved Ontology consists of a owl file and a pkl file with all the
        data (class2idx, idx2labs, inv_idx, ...) generated from construction
        """
        Path(saved_path).mkdir(parents=True, exist_ok=True)
        self.copy2(self.owl_path, saved_path)
        owl_file_name = self.owl_path.split("/")[-1]
        self.owl_path = saved_path + "/" + owl_file_name
        # the owlready2 ontology cannot be pickled, so saved as a separate file
        delattr(self, "owl")
        self.save_pkl(self, saved_path)
        # load back the owl ontology after saving the pickled parts
        self.owl = get_ontology(f"file://{self.owl_path}").load()

    def __str__(self) -> str:
        self.info = AttrDict(
            {
                "owl_name": self.owl.name,
                "num_classes": self.num_classes,
                "lab_probs": self.lab_props,
                "num_labs": self.num_labs,
                "avg_labs": self.avg_labs,
                "num_entries_inv_idx": self.num_entries_inv_idx,
            }
        )
        return super().report(**self.info)
    
    # def destroy_owl_cache(self):
    #     """Owlready2 does *not* create a new object when IRI coincide, to make sure we are 
    #     operating on the correct owl object, we need to destroy the previous cached entities
    #     """
    #     self.owl._destroy_cached_entities()
        
    # def reload_onto_without_inv_idx(self):
    #     """Destroy the previous cached entities and reload the owl object
    #     """
    #     self.destroy_owl_cache()
    #     return Ontology.from_new(self.owl_path, self.lab_props)

    @staticmethod
    def assign_class_numbers(owl_onto):
        """Assign numbers for each class in an owlready2 ontology
        """
        cl_iris = [text_utils.abbr_iri(cl.iri) for cl in owl_onto.classes()]
        cl_idx = list(range(len(cl_iris)))
        class2idx = dict(zip(cl_iris, cl_idx))
        idx2class = dict(zip(cl_idx, cl_iris))
        assert len(class2idx) == len(idx2class)
        return class2idx, idx2class

    def build_inv_idx(self, tokenizer, cut: int = 0) -> None:
        """Create inverted index based on the extracted labels of an ontology

        Args:
            tokenizer : text tokenizer, word-level or sub-word-level,
                        a .tokenize() funciton needs to be implemented
            cut (int): keep tokens with length > cut 
        """
        self.inv_idx = defaultdict(list)
        for cls_iri, cls_labs in self.idx2labs.items():
            for tk in tokenizer.tokenize_all(cls_labs):
                if len(tk) > cut:
                    self.inv_idx[tk].append(cls_iri)
        self.num_entries_inv_idx = len(self.inv_idx)

    def idf_select(self, ent_toks, pool_size: int = 200) -> List[str]:
        """Select entities based on idf scores
        """
        cand_pool = text_utils.idf_select(ent_toks, self.inv_idx, pool_size)
        # print three selected examples
        examples = [
            (self.idx2class[ent_id], round(idf_score, 1))
            for ent_id, idf_score in cand_pool[: min(pool_size, 3)]
        ]
        info = {"num_candidates": len(cand_pool)}
        for i in range(len(examples)):
            info[f"example_{i+1}"] = examples[i]
        print(self.report(root_name="Selection.Info", **info))
        return cand_pool

    def path_select(self, pool_size: int = 200) -> List[str]:
        # TODO
        pass