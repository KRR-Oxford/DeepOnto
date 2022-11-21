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
"""Class that extends the owlready2 Ontology for convenient access to textual and structural information

NOTE: the logical information is handled by the OWLAPI library (see onto.logic.reasoner)

Known issues for owlready2:
- No new object will be created when loading ontologies of the same IRIs 
    - Solution: beforing handling the second owl of same IRI, destroy the cache in the first one

"""

from __future__ import annotations

import os
from typing import Optional, List
from collections import defaultdict
from owlready2 import get_ontology, default_world
from owlready2.entity import ThingClass
from pathlib import Path

from deeponto import SavedObj
from .text import Tokenizer, text_utils


class Ontology(SavedObj):
    def __init__(self, owl_path: str):
        # owlready2 attributes
        self.owl_path = os.path.abspath(owl_path)
        self.owl = get_ontology(f"file://{owl_path}").load()
        # self.graph = default_world.as_rdflib_graph()  # rdf graph
        # NOTE: list of label properties (in IRIs)
        self.lab_props = None
        # entity or property IRIs to their labels (via specified annotation properties)
        self.iri2labs = None
        # inverted index for class labels
        self.inv_idx = None

        # stat attributes
        self.num_classes = None
        self.num_labs = None
        self.avg_labs = None
        self.num_entries_inv_idx = None
        super().__init__(self.owl.name)

    @classmethod
    def from_new(
        cls,
        onto_path: str,
        lab_props: List[str] = ["http://www.w3.org/2000/01/rdf-schema#label"],
        tokenizer: Optional[Tokenizer] = None,
        uncased_labels: bool = True,
    ) -> Ontology:
        """Initialise a new Ontology instance and coduct pre-processing

        Parameters
        ----------
        onto_path : str
            path to the ontology file to be processed
        lab_props : List[str], optional
            annotation properties considered as the class labels, by default ["http://www.w3.org/2000/01/rdf-schema#label"]
        tokenizer : Optional[Tokenizer], optional
            tokenizer used for creating the inverted index, by default None
        uncased_labels : bool, optional
            lowercased the pre-processed class labels or not, by default True

        Returns
        -------
        Ontology
            a pre-processed Ontology instance
        """
        onto = cls(onto_path)
        onto.lab_props = lab_props
        onto.iri2labs = defaultdict(list)
        onto.num_classes = 0
        onto.num_labs = 0
        for cl in onto.owl.classes():
            onto.iri2labs[cl.iri] = text_utils.labs_from_props(
                cl.iri, onto.lab_props, uncased_labels
            )
            onto.num_labs += len(onto.iri2labs[cl.iri])
            onto.num_classes += 1
        onto.avg_labs = round(onto.num_labs / onto.num_classes, 2)
        # for (sub-)word inverted index
        if tokenizer:
            onto.build_inv_idx(tokenizer, cut=0)
        print(onto)
        return onto

    @classmethod
    def from_saved(cls, saved_path: str) -> Optional[Ontology]:
        """Load an instance of pre-processed ontologies from the specified path.

        Parameters
        ----------
        saved_path : str
            path to save the pre-processed ontology

        Returns
        -------
        Optional[Ontology]
            loaded pre-processed ontology

        Raises
        ------
        FileNotFoundError
            raise if the required files are missing
        """
        try:
            onto = cls.load_pkl(saved_path)
            owl_file_name = onto.owl_path.split("/")[-1]
            onto.owl_path = saved_path + "/" + owl_file_name
            onto.owl = get_ontology(f"file://{onto.owl_path}").load()
            return onto
        except:
            raise FileNotFoundError(f"please check file integrity in : {saved_path})")

    def save_instance(self, saved_path: str):
        """Save the current instance of an pre-processed ontologies
        
        The saved Ontology consists of a owl file and a pkl file with all the
        data (class2idx, idx2labs, inv_idx, ...) generated from construction

        Parameters
        ----------
        saved_path : str
            path to save the Ontology instance
        """
        Path(saved_path).mkdir(parents=True, exist_ok=True)
        self.copy2(self.owl_path, saved_path)
        owl_file_name = self.owl_path.split("/")[-1]
        self.owl_path = saved_path + "/" + owl_file_name
        # the owlready2 ontology cannot be pickled, so saved as a separate file
        delattr(self, "owl")
        # delattr(self, "graph")
        self.save_pkl(self, saved_path)
        # load back the owl ontology after saving the pickled parts
        self.owl = get_ontology(f"file://{self.owl_path}").load()

    def __str__(self) -> str:
        self.info = {
            "owl_name": self.owl.name,
            "num_classes": self.num_classes,
            "lab_probs": [self.name_from_iri(p) for p in self.lab_props if self.obj_from_iri(p)],
            "num_labs": self.num_labs,
            "avg_labs": self.avg_labs,
            "num_entries_inv_idx": self.num_entries_inv_idx,
        }

        return super().report(**self.info)

    @property
    def classes(self):
        return list(self.owl.classes())

    @property
    def iri(self):
        return self.owl.base_iri

    @classmethod
    def obj_from_iri(cls, iri: str):
        """Return an owlready2 entity given its IRI
        
        Parameters
        ----------
        iri : str
            the IRI of the entity

        Returns
        -------
        owlready2.Entity
            the owlready2 entity that has the specified IRI
        """
        return default_world[iri]

    @classmethod
    def name_from_iri(cls, iri: str):
        """Return the entity name of the specified IRI

        Parameters
        ----------
        iri : str
            the IRI of the owlready2 entity

        Returns
        -------
        str
            the entity name of the specified IRI
        """
        return str(cls.obj_from_iri(iri))

    def search_ent_labs(self, ent: ThingClass):
        """Search labels for a given owlready2 ThingClass

        Parameters
        ----------
        ent : ThingClass
            the ThingClass object to search for labels

        Returns
        -------
        list
            the labels for the given ThingClass

        Raises
        ------
        ValueError
            raise if the given ThingClass is not found
        """
        if ent.iri in self.iri2labs.keys():
            return self.iri2labs[ent.iri]
        else:
            raise ValueError(f'Input entity class "{ent.iri}" is not found in the ontology ...')

    def sib_labs(self):
        """Return all the label groups extracted from sibling classes of this ontology as a 3-D list:

        Returns
        -------
        List[List[List]]
            -   1st list for different sibling groups;
            -   2nd list for different siblings;
            -   3rd list for different labels.
        """
        return text_utils.sib_labs(self.classes, self.lab_props)

    def build_inv_idx(self, tokenizer, cut: int = 0) -> None:
        """Create inverted index based on the extracted labels of an ontology

        Parameters
        ----------
        tokenizer : Tokenizer
            text tokenizer, word-level or sub-word-level
        cut : int, optional
            keep tokens with length > cut , by default 0
        """
        self.inv_idx = defaultdict(list)
        for cls_iri, cls_labs in self.iri2labs.items():
            for tk in tokenizer.tokenize_all(cls_labs):
                if len(tk) > cut:
                    self.inv_idx[tk].append(cls_iri)
        self.num_entries_inv_idx = len(self.inv_idx)

    def idf_select(self, ent_toks, pool_size: int = 200) -> List[str]:
        """Select entities based on idf scores calculated from the pre-computed inverted index

        Parameters
        ----------
        ent_toks : List[str]
            tokenized entity labels
        pool_size : int, optional
            maximum number of candidates considered, by default 200

        Returns
        -------
        List[str]
            candidates ranked by idf scores
        """
        cand_pool = text_utils.idf_select(ent_toks, self.inv_idx, pool_size)
        # print three selected examples
        examples = [
            (self.name_from_iri(ent_iri), round(idf_score, 1))
            for ent_iri, idf_score in cand_pool[: min(pool_size, 3)]
        ]
        info = {"num_candidates": len(cand_pool)}
        for i in range(len(examples)):
            info[f"example_{i+1}"] = examples[i]
        print(self.report(root_name="Selection.Info", **info))
        return cand_pool

    def path_select(self, pool_size: int = 200) -> List[str]:
        # TODO
        pass
