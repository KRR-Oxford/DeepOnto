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
"""Script for BERTMap's Text Semantics Corpora"""

from __future__ import annotations

from typing import Optional, List, Union
from pyats.datastructures import AttrDict

from deeponto.onto.text import Thesaurus, text_utils
from deeponto.onto import Ontology
from deeponto.onto.mapping import OntoMappings
from deeponto import SavedObj
from deeponto.utils import uniqify
from deeponto.utils.logging import banner_msg


class TextSemanticsCorpora(SavedObj):
    def __init__(
        self,
        src_onto: Ontology,
        tgt_onto: Ontology,
        known_mappings: Optional[OntoMappings] = None,
        aux_ontos: Optional[List[Ontology]] = None,
        apply_transitivity: bool = False,
        neg_ratio: int = 4,
    ):
        self.src_onto = src_onto
        self.tgt_onto = tgt_onto
        self.known_mappings = known_mappings
        # list of auxiliary ontologies
        self.aux_ontos = aux_ontos
        self.apply_transitivity = apply_transitivity
        self.neg_ratio = neg_ratio
        self.positives = []
        self.soft_negatives = []
        self.hard_negatives = []

        self.stats = AttrDict(
            {
                "transitivie": self.apply_transitivity,
                "num_known_mappings": len(self.known_mappings.to_tuples())
                if self.known_mappings
                else "N/A",
                "num_aux_ontos": len(self.aux_ontos) if self.aux_ontos else "N/A",
            }
        )  # storing the statistics of corpora

        # build thesaurus and then text semantics sub-corpora from various sources
        self.thesaurus = Thesaurus(self.apply_transitivity)

        # intra-onto
        self.intra_onto_corpus_src = TextSemanticsCorpusforOnto(
            self.src_onto, self.thesaurus, self.neg_ratio, flag="src"
        )
        self.add_samples_from_sub_corpus(self.intra_onto_corpus_src)

        self.intra_onto_corpus_tgt = TextSemanticsCorpusforOnto(
            self.tgt_onto, self.thesaurus, self.neg_ratio, flag="tgt"
        )
        self.add_samples_from_sub_corpus(self.intra_onto_corpus_tgt)

        # cross-onto
        self.cross_onto_corpus = None
        if self.known_mappings:
            self.cross_onto_corpus = TextSemanticsCorpusforMappings(
                self.src_onto, self.tgt_onto, self.known_mappings, self.thesaurus, self.neg_ratio
            )
            self.add_samples_from_sub_corpus(self.cross_onto_corpus)

        # complementary
        self.comple_corpora = []
        if self.aux_ontos:
            for aux_onto in self.aux_ontos:
                aux_corpus = TextSemanticsCorpusforOnto(
                    aux_onto, self.thesaurus, self.neg_ratio, flag="aux"
                )
                self.comple_corpora.append(aux_corpus)
                self.add_samples_from_sub_corpus(aux_corpus)

        if self.apply_transitivity:
            # copy and mark the individually retrieved samples as "isolated"
            self.positives_isolated = self.positives
            self.soft_negatives_isolated = self.soft_negatives
            # self.hard_negatives_isolated = self.hard_negatives
            # only the merged section is needed in this case because all synonym groups are merged by transitivity
            self.positives = Thesaurus.positive_sampling(self.thesaurus.merged_section)
            # however, the hard negatives rely on individual ontology structure, so we keep the hard negatives
            # and amend more soft negatives
            total_neg_num = self.neg_ratio * len(self.positives)
            self.soft_negatives = Thesaurus.random_negative_sampling(
                self.thesaurus.merged_section, neg_num=total_neg_num - len(self.hard_negatives)
            )

        self.positives = uniqify(self.positives)
        self.soft_negatives = uniqify(self.soft_negatives)
        self.hard_negatives = uniqify(self.hard_negatives)
        # now we combine the soft and hard negatives and remove duplicates
        self.negatives = uniqify(self.soft_negatives + self.hard_negatives)
        # we then remove the invalid negatives (which appear in the positives)
        self.negatives = self.remove_invalid_negatives(self.positives, self.negatives)
        # TODO: the number of negatives does not strictly follow the ratio after uniqifying
        # TODO: maybe it is better to down sample the negatives again here before fine-tuning
        # TODO: will leave for future improvement

        # record the stats
        self.stats.num_positives = len(self.positives)
        self.stats.num_negatives = len(self.negatives)
        self.stats.soft_negatives = len(self.soft_negatives)
        self.stats.num_hard_negatives = len(self.hard_negatives)

        super().__init__(f"txtsem.corpora")

    def __str__(self) -> str:
        return super().report(**self.stats)

    def report_sub_corpora_info(self):
        """Print individual sub-corpora information
        """
        banner_msg("Text Semantics Corpora (Overall)")
        print(self)
        banner_msg("Intra-onto Corpus from Source Ontology")
        print(str(self.intra_onto_corpus_src))
        banner_msg("Intra-onto Corpus from Target Ontology")
        print(str(self.intra_onto_corpus_tgt))
        if self.cross_onto_corpus:
            banner_msg("Cross-onto Corpus from Src2Tgt Mappings")
            print(str(self.cross_onto_corpus))
        if self.comple_corpora:
            banner_msg("Complemenatry Corpora from Auxiliary Ontologies")
            for cp_corpus in self.comple_corpora:
                print(str(cp_corpus))

    def add_samples_from_sub_corpus(
        self, sub_corpus: Union[TextSemanticsCorpusforOnto, TextSemanticsCorpusforMappings]
    ):
        """Add positives and negatives into the final collections
        """
        self.positives += sub_corpus.positives
        self.soft_negatives += sub_corpus.soft_negatives
        self.hard_negatives += sub_corpus.hard_negatives

    @staticmethod
    def remove_invalid_negatives(pos_samples: List, neg_samples: List):
        """Remove elements in negative samples that have been occurred in positive samples
        """
        print("Removing invalid negatives that have been occurred in positive samples ...")
        pos_set = set(pos_samples)
        neg_set = set(neg_samples)
        intersection = pos_set.intersection(neg_set)
        cleaned_neg_samples = list(neg_set - intersection)
        print(
            f"\t[before]={len(neg_samples)}; [after]={len(cleaned_neg_samples)}; [removed]={len(intersection)}"
        )
        return cleaned_neg_samples


class TextSemanticsCorpusforOnto(SavedObj):
    """Class for contructing text semantics Corpora from an individual ontology
    """

    def __init__(self, onto: Ontology, thesaurus: Thesaurus, neg_ratio: int = 4, flag: str = "src"):
        self.onto = onto
        self.thesaurus = thesaurus
        self.neg_ratio = neg_ratio
        self.flag = flag  # can be "src", "tgt" or "aux"
        # extract synonyms from individual onto and add it into the thesaurus
        self.synonym_field = self.thesaurus.add_synonyms_from_ontos(onto)
        self.positives = Thesaurus.positive_sampling(self.synonym_field)
        # we sample for each positive sample R negative samples
        total_neg_num = self.neg_ratio * len(self.positives)
        # hard negatives are not always available so we compute them first
        # ideally they should be half of the negatives
        self.hard_negatives = Thesaurus.disjointness_negative_sampling(
            text_utils.sib_labs(self.onto.owl.classes()), neg_num=(total_neg_num // 2)
        )
        # soft negatives are almost always available so we amend more if we do not have enough negatives
        self.soft_negatives = Thesaurus.random_negative_sampling(
            self.synonym_field, neg_num=(total_neg_num - len(self.hard_negatives))
        )
        self.negatives = self.soft_negatives + self.hard_negatives
        
        self.stats = {
            "onto": self.onto.owl.name,
            "flag": self.flag,
            "num_positives": len(self.positives),
            "num_negatives": len(self.negatives),
            "num_soft_negatives": len(self.soft_negatives),
            "num_hard_negatives": len(self.hard_negatives),
        }
        
        super().__init__(f"txtsem.corpus.onto")

    def __str__(self):
        info = "---------- Ontology ----------\n" + str(self.onto) + "\n"
        info += "---------- Intra-onto Corpus ----------\n"+ self.report(**self.stats)
        return info


class TextSemanticsCorpusforMappings(SavedObj):
    def __init__(
        self,
        src_onto: Ontology,
        tgt_onto: Ontology,
        known_mappings: OntoMappings,
        thesaurus: Thesaurus,
        neg_ratio: int = 4,
    ):
        self.src_onto = src_onto
        self.tgt_onto = tgt_onto
        self.known_mappings = known_mappings
        self.thesaurus = thesaurus
        self.neg_ratio = neg_ratio
        self.flag = "src2tgt"

        # extract synonyms from mappings and add it into the thesaurus
        self.cross_onto_synonym_field = thesaurus.add_matched_synonyms_from_mappings(
            self.src_onto, self.tgt_onto, self.known_mappings
        )
        self.positives = Thesaurus.positive_sampling_from_paired_groups(
            self.cross_onto_synonym_field
        )
        # we sample for each positive sample R negative samples
        total_neg_num = self.neg_ratio * len(self.positives)
        # only soft negatives are available at cross-onto level
        self.soft_negatives = Thesaurus.random_negative_sampling_from_paired_groups(
            self.cross_onto_synonym_field, neg_num=total_neg_num
        )
        self.hard_negatives = []
        self.negatives = self.soft_negatives
        
        self.stats = {
            "src_onto": self.src_onto.owl.name,
            "tgt_onto": self.tgt_onto.owl.name,
            "flag": self.flag,
            "num_positives": len(self.positives),
            "num_negatives": len(self.negatives),
            "num_known_mappings": len(self.known_mappings.to_tuples())
        }

        super().__init__(f"txtsem.corpus.maps")

    def __str__(self):
        info = "---------- SRC Ontology ----------\n" + str(self.src_onto) + "\n"
        info += "---------- TGT Ontology ----------\n" + str(self.tgt_onto) + "\n"
        info += "---------- Cross-onto Corpus ----------\n"+ self.report(**self.stats)
        return info
