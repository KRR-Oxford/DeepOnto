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
"""Class for BERTMap

*An importance notice*: to avoid that the auxiliary ontology might have the same IRI 
as the SRC or TGT ontologies so that OwlReady2 cannot distinguish them, we load aux ontos
only after we have built (intra-onto / cross-onto) corpora for SRC and TGT ontologies

*Another interesting fact*: since cross-onto corpora do not depend on the owl object after
parsing into our Ontology class, so even destroying the SRC and TGT owls will not make a 
difference for creating validation and testing mapping corpora

"""

import os
import itertools
import torch
from typing import Optional, List
from pyats.datastructures import AttrDict
from sklearn.model_selection import train_test_split

from deeponto.bert import BERTArgs
from deeponto.bert.static import BERTStaticSeqClassifer
from deeponto.bert.tune import BERTFineTuneSeqClassifier
from deeponto.onto.text import Tokenizer
from deeponto.onto.text.thesaurus import Thesaurus
from deeponto.onto import Ontology
from deeponto.onto.mapping import OntoMappings
from deeponto.utils import detect_path, create_path
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
        apply_string_match: bool = True,
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
        self.val_mappings = validation_mappings
        self.test_mappings = test_mappings
        self.aux_ontos = aux_ontos
        self.apply_transitivity = apply_transitivity
        self.neg_ratio = neg_ratio
        self.apply_string_match = apply_string_match

        # text semantics corpora
        self.corpora_path = self.saved_path + "/corpora"
        self.main_corpora, self.val_maps_corpus, self.test_maps_corpus = None, None, None
        self.construct_corpora()

        # fine-tuning data from corpora
        self.fine_tune_data_path = self.saved_path + "/fine_tune/data"
        self.fine_tune_data = None
        self.load_fine_tune_data(split_ratio=0.1)

        # BERT model initialization
        self.fine_tune_model_path = self.saved_path + "/fine_tune/model"
        self.bert_classifier = None

    def train(self):
        """BERT synonym classifier fine-tuning
        """
        banner_msg("Fine-tune BERT Synonym Classifier")
        if not detect_path(self.fine_tune_model_path) or self.bert_args.resume_from_ckp:
            if not detect_path(self.fine_tune_model_path):
                create_path(self.fine_tune_model_path)
            if not self.bert_args.resume_from_ckp:
                print("Start training from scratch ...")
            else:
                print("Resume training from previous checkpoint...")
            bert_classifier = BERTFineTuneSeqClassifier(
                bert_args=self.bert_args,
                train_data=self.fine_tune_data.train,
                val_data=self.fine_tune_data.val,
                test_data=self.fine_tune_data.test,
            )
            bert_classifier.train()
            if bert_classifier.test_data:
                print("Generate synonym classification results on intermediate testing set ...")
                intermediate_test_results = bert_classifier.trainer.evaluate(
                    bert_classifier.test_data
                )
                data_size_string = f"{bert_classifier.train_size}-{bert_classifier.val_size}-{bert_classifier.test_size}"
                intermediate_test_results["train-val-test_sizes"] = data_size_string
                SavedObj.print_json(intermediate_test_results)
                SavedObj.save_json(
                    intermediate_test_results,
                    self.fine_tune_model_path + "/interm_test_results.json",
                )
        else:
            print("found an existing BERT model directory, delete it and re-run if it's empty ...")
            best_checkpoint = 0
            for file in os.listdir(self.fine_tune_model_path):
                # only 1 checkpoint is saved so the latest one is the best
                if file.startswith("checkpoint"):
                    trainer_state = SavedObj.load_json(
                        f"{self.fine_tune_model_path}/{file}/trainer_state.json"
                    )
                    checkpoint = int(
                        trainer_state["best_model_checkpoint"].split("/")[-1].split("-")[-1]
                    )
                    if checkpoint > best_checkpoint:
                        best_checkpoint = checkpoint
                        print(f"found new checkpoint: {best_checkpoint} ...")
            banner_msg(f"Found Saved Best Checkpoint: {best_checkpoint}")
            self.bert_args.bert_checkpoint = (
                f"{self.fine_tune_model_path}/checkpoint-{best_checkpoint}"
            )
        self.bert_classifier = BERTStaticSeqClassifer(self.bert_args)

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
            self.aux_ontos = text_semantics_corpora.aux_ontos
            print("Save the main corpora data and construction report ...")
            
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

    def string_match(self, src_ent_id: str, tgt_ent_id: str):
        """Predict `easy` mappings by applying string-matching
        """
        src_ent_labs = self.src_onto.idx2labs[src_ent_id]
        tgt_ent_labs = self.tgt_onto.idx2labs[tgt_ent_id]
        overlap_labs = set(src_ent_labs).intersection(set(tgt_ent_labs))
        return int(len(overlap_labs) > 0)

    def ent_pair_score(self, src_ent_id: str, tgt_ent_id: str):
        """Compute mapping score between a cross-ontology entity pair
        """
        src_ent_labs = self.src_onto.idx2labs[src_ent_id]
        tgt_ent_labs = self.tgt_onto.idx2labs[tgt_ent_id]
        # apply string-match before using bert
        if self.apply_string_match:
            prelim_score = self.string_match(src_ent_labs, tgt_ent_labs)
            if prelim_score == 1.0:
                return prelim_score
        # apply BERT classifier and define mapping score := Average(SynonymScores)
        src_tgt_lab_product = itertools.product(src_ent_labs, tgt_ent_labs)
        # only one element tensor is able to be extracted as a scalar by .item()
        return torch.mean(self.bert_classifier(src_tgt_lab_product)).item()

    def global_mappings_for_ent(self, src_ent_id: int):
        """Compute cross-ontology mappings for a source entity
        """
        mappings_for_ent = super().global_mappings_for_ent(src_ent_id)
        # get source entity contents
        src_ent_name = self.src_onto.idx2class[src_ent_id]
        # select target candidates and compute score for each
        tgt_cands = self.idf_select_for_ent(src_ent_id)  # [(tgt_id, idf_score)]

        # for string_match
        if self.apply_string_match:
            for tgt_cand_id, _ in tgt_cands:
                tgt_ent_name = self.tgt_onto.idx2class[tgt_cand_id]
                mapping_score = self.string_match(src_ent_id, tgt_cand_id)
                if mapping_score > 0:
                    # save mappings only with positive mapping scores
                    mappings_for_ent.append(
                        self.set_mapping(src_ent_name, tgt_ent_name, mapping_score)
                    )
                    # output only the top (k=n_best) scored mappings
            # return the mappings if there are any string match results
            if len(mappings_for_ent) > 0:
                n_best_mappings_for_ent = mappings_for_ent.top_k(self.n_best)
                self.logger.info(f"[{self.flag}: {src_ent_id}] {n_best_mappings_for_ent}\n")
                return n_best_mappings_for_ent

        # for bert synonym classifier
        labs_batches = self.batched_lab_products_for_ent(
            src_ent_id, tgt_cands, self.bert_args.batch_size_for_prediction
        )  # [{"labs": [], "lens": []}]
        batch_base_idx = 0  # after each batch, the base index will be increased by # of covered target candidates
        n_best_scores = torch.tensor([-1] * self.n_best).to(self.bert_classifier.device)
        n_best_idxs = torch.tensor([-1] * self.n_best).to(self.bert_classifier.device)

        for labs_batch in labs_batches:
            batch_scores = self.bert_classifier(labs_batch.labs)
            pooled_scores = self.batch_pooling(batch_scores, labs_batch.lens)
            # K should be n_best, except when the pooled batch scores do not contain K values
            K = len(pooled_scores) if len(pooled_scores) < self.n_best else self.n_best
            batch_top_k_scores, batch_top_k_idxs = torch.topk(pooled_scores, k=K)
            batch_top_k_idxs += batch_base_idx
            # we do the substitution for every batch to prevent from memory overflow
            n_best_scores, best_scores_idxs = torch.topk(
                torch.cat([batch_top_k_scores, n_best_scores]), k=self.n_best
            )
            n_best_idxs = torch.cat([batch_top_k_idxs, n_best_idxs])[best_scores_idxs]
            # print(f"current nbest idx: {batch_nbest_idxs}")
            batch_base_idx += len(
                labs_batch.lens
            )  # num of lens is exactly the num of tgt candidates in this batch

        for idx, score in zip(n_best_idxs, n_best_scores):
            # ignore too small values or intial values (-1.0) for dummy mappings
            if score.item() >= 0.01:
                tgt_ent_name = self.tgt_onto.idx2class[tgt_cands[idx.item()][0]]
                mappings_for_ent.append(self.set_mapping(src_ent_name, tgt_ent_name, score.item()))

        # output only the top (k=n_best) scored mappings
        n_best_mappings_for_ent = mappings_for_ent.top_k(self.n_best)
        self.logger.info(f"[{self.flag}: {src_ent_id}] {n_best_mappings_for_ent}\n")
        return n_best_mappings_for_ent

    def batch_pooling(self, batch_scores: torch.Tensor, batch_lens: List[int]) -> torch.Tensor:
        """Split the tensors by specified lengths and compute the mean for each part
        """
        seq_of_scores = torch.split(batch_scores, split_size_or_sections=batch_lens)
        pooled_batch_scores = [torch.mean(chunk) for chunk in seq_of_scores]
        return torch.stack(pooled_batch_scores)
