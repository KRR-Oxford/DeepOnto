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
"""Class for BERTMap"""

from typing import Optional, List, Set
from textdistance import levenshtein
from itertools import product
from pyats.datastructures import AttrDict
from sklearn.model_selection import train_test_split

from deeponto.bert import BERTArgs
from deeponto.bert.tune import BERTFineTuneSeqClassifier
from deeponto.onto.text import Tokenizer
from deeponto.onto.text.thesaurus import Thesaurus
from deeponto.onto import Ontology
from deeponto.onto.mapping import OntoMappings
from deeponto.utils import detect_path, create_path, uniqify
from deeponto.utils.logging import banner_msg
from deeponto import SavedObj
from .corpora import TextSemanticsCorpora, TextSemanticsCorpusforMappings
from .. import OntoAlign


class BERTMap(OntoAlign):
    def __init__(
        self,
        src_onto: Ontology,
        tgt_onto: Ontology,
        tokenizer: Tokenizer,
        bert_args: BERTArgs,  # arguments for BERT fine-tuning
        cand_pool_size: Optional[int] = 200,
        n_best: Optional[int] = 10,
        saved_path: str = "",
        train_mappings: Optional[OntoMappings] = None,  # cross-ontology corpus if provided
        validation_mappings: Optional[OntoMappings] = None,  # for validation
        test_mappings: Optional[OntoMappings] = None,  # TODO: we may not need the testing data
        aux_ontos: List[Ontology] = [],  # complementary corpus if provided
        apply_transitivity: bool = False,  # obtain more synonyms/non-synonyms by applying transitivity?
        neg_ratio: int = 4,
    ):
        super().__init__(
            src_onto=src_onto,
            tgt_onto=tgt_onto,
            tokenizer=tokenizer,
            cand_pool_size=cand_pool_size,
            rel="≡",
            n_best=n_best,
            is_trainable=True,
            saved_path=saved_path,
        )
        self.bert_args = bert_args
        self.known_mappings = train_mappings
        self.aux_ontos = aux_ontos
        self.apply_transitivity = apply_transitivity
        self.neg_ratio = neg_ratio
        self.val_mappings = validation_mappings
        self.test_mappings = test_mappings

        # text semantics corpora
        self.corpora_path = self.saved_path + "/corpora"
        self.main_corpora, self.val_maps_corpus, self.test_maps_corpus = None, None, None
        self.construct_corpora()

        # fine-tuning data from corpora
        self.fine_tune_data_path = self.saved_path + "/fine_tune/data"
        self.fine_tune_data = None
        self.load_fine_tune_data(split_ratio=0.1)

    def construct_corpora(self):
        """Load corpora data from new construction or saved directory
        """
        # Text Semantics Corpora
        banner_msg("Text Semantics Corpora")
        if not detect_path(self.corpora_path):
            print("Create text semantics corpora for *train-val* in fine-tuning ...")
            text_semantics_corpora = TextSemanticsCorpora(
                src_onto=self.src_onto,
                tgt_onto=self.tgt_onto,
                known_mappings=self.known_mappings,
                aux_ontos=self.aux_ontos,
                apply_transitivity=self.apply_transitivity,
                neg_ratio=self.neg_ratio,
            )
            text_semantics_corpora.save_instance(self.corpora_path, flag="train-val")
            print("Save the training corpora data and construction report ...")
            if self.val_mappings:
                print("Create text semantics corpora for *val* (from mappings) in fine-tuning ...")
                validation_corpus = TextSemanticsCorpusforMappings(
                    src_onto=self.src_onto,
                    tgt_onto=self.tgt_onto,
                    onto_mappings=self.val_mappings,
                    thesaurus=Thesaurus(apply_transitivity=self.apply_transitivity),
                )
                validation_corpus.save_instance(self.corpora_path, flag="val.maps")
                print("Save the validation corpora data and construction report ...")
            if self.test_mappings:
                print(
                    "Create text semantics corpora for *testing* (from mappings) in fine-tuning ..."
                )
                testing_corpus = TextSemanticsCorpusforMappings(
                    src_onto=self.src_onto,
                    tgt_onto=self.tgt_onto,
                    onto_mappings=self.test_mappings,
                    thesaurus=Thesaurus(apply_transitivity=self.apply_transitivity),
                )
                testing_corpus.save_instance(self.corpora_path, flag="test.maps")
                print("Save the testing corpora data and construction report ...")
        else:
            print("found an existing corpora directory, delete it and re-run if it's empty ...")
            print("if constructed, check details in `report.txt` ...")
        print("Loading the constructed corpora data ...")
        self.main_corpora = AttrDict(SavedObj.load_json(self.corpora_path + "/train-val.json"))
        banner_msg("Corpora Statistics (Train-Val)")
        SavedObj.print_json(self.main_corpora.stats)
        if detect_path(self.corpora_path + "/val.maps.json"):
            self.val_maps_corpus = AttrDict(
                SavedObj.load_json(self.corpora_path + "/val.maps.json")
            )
            banner_msg("Corpora Statistics (Val-Maps)")
            SavedObj.print_json(self.val_maps_corpus.stats)
        if detect_path(self.corpora_path + "/test.maps.json"):
            self.test_maps_corpus = AttrDict(
                SavedObj.load_json(self.corpora_path + "/test.maps.json")
            )
            banner_msg("Corpora Statistics (Test-Maps)")
            SavedObj.print_json(self.test_maps_corpus.stats)

    def load_fine_tune_data(self, split_ratio: float = 0.1):
        """Get data for fine-tuning from the corpora
        """
        banner_msg("Fine-tuning Data")
        if not detect_path(self.fine_tune_data_path):
            fine_tune_data = AttrDict()
            fine_tune_data.stats = dict()
            print(
                f"Splitting main corpora into training and validation ({split_ratio * 100}%) data ..."
            )
            main_data = self.main_corpora.positives + self.main_corpora.negatives
            main_train, main_val = train_test_split(main_data, test_size=split_ratio)
            fine_tune_data.train = main_train
            fine_tune_data.val = main_val
            fine_tune_data.test = []
            if self.val_maps_corpus:
                print("Get additional validation data from validation mappings ...")
                # TODO: we do not care about duplicates here because label pairs from mappings are of higher importance
                fine_tune_data.val += (
                    self.val_maps_corpus.positives + self.val_maps_corpus.negatives
                )
            if self.test_maps_corpus:
                print("Get additional testing data from testing mappings ...")
                print(
                    "\t=> These testing mapppings do not make any decision on model selection ..."
                )
                fine_tune_data.test += (
                    self.test_maps_corpus.positives + self.test_maps_corpus.negatives
                )
            fine_tune_data.stats["n_train"] = len(fine_tune_data.train)
            fine_tune_data.stats["n_val"] = len(fine_tune_data.val)
            fine_tune_data.stats["n_test"] = len(fine_tune_data.test)
            create_path(self.fine_tune_data_path)
            SavedObj.save_json(fine_tune_data, self.fine_tune_data_path + "/data.json")
        else:
            print(
                "found an existing fine-tune data directory, delete it and re-run if it's empty ..."
            )
        self.fine_tune_data = AttrDict(SavedObj.load_json(self.fine_tune_data_path + "/data.json"))
        SavedObj.print_json(self.fine_tune_data.stats)

    def ent_pair_score(self, src_ent_id: str, tgt_ent_id: str):
        """Compute mapping score between a cross-ontology entity pair
        """
        src_ent_labs = self.src_onto.idx2labs[src_ent_id]
        tgt_ent_labs = self.tgt_onto.idx2labs[tgt_ent_id]
        if not self.use_edit_dist:
            mapping_score = int(len(self.overlap(src_ent_labs, tgt_ent_labs)) > 0)
        else:
            mapping_score = self.max_norm_edit_sim(src_ent_labs, tgt_ent_labs)
        return mapping_score

    def global_mappings_for_ent(self, src_ent_id: int):
        """Compute cross-ontology mappings for a source entity
        """
        mappings_for_ent = super().global_mappings_for_ent(src_ent_id)
        # get source entity contents
        src_ent_name = self.src_onto.idx2class[src_ent_id]
        # select target candidates and compute score for each
        tgt_cands = self.idf_select_for_ent(src_ent_id)
        for tgt_cand_id, _ in tgt_cands:
            tgt_ent_name = self.tgt_onto.idx2class[tgt_cand_id]
            mapping_score = self.ent_pair_score(src_ent_id, tgt_cand_id)
            if mapping_score > 0:
                # save mappings only with positive mapping scores
                mappings_for_ent.append(self.set_mapping(src_ent_name, tgt_ent_name, mapping_score))
        # output only the top (k=n_best) scored mappings
        n_best_mappings_for_ent = mappings_for_ent.top_k(self.n_best)
        self.logger.info(f"[{self.flag}: {src_ent_id}] {n_best_mappings_for_ent}\n")
        return n_best_mappings_for_ent

    @staticmethod
    def overlap(src_ent_labs: List[str], tgt_ent_labs: List[str]) -> Set:
        # TODO: the overlapped percentage could be a factor of judgement
        return set(src_ent_labs).intersection(set(tgt_ent_labs))

    @classmethod
    def max_norm_edit_sim(cls, src_ent_labs: List[str], tgt_ent_labs: List[str]) -> float:
        # save time from the special case of overlapped labels
        if cls.overlap(src_ent_labs, tgt_ent_labs):
            return 1.0
        label_pairs = product(src_ent_labs, tgt_ent_labs)
        sim_scores = [levenshtein.normalized_similarity(src, tgt) for src, tgt in label_pairs]
        return max(sim_scores)
