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

from __future__ import annotations

import os
import itertools
import torch
from typing import Optional, List
from sklearn.model_selection import train_test_split
import subprocess

import deeponto
from deeponto.bert import BertArguments
from deeponto.bert.static import BertStaticForSequenceClassification
from deeponto.bert.train import BertTrainerForSequenceClassification
from deeponto.onto.text import Tokenizer, Thesaurus
from deeponto.data.align import (
    TextSemanticsCorpora,
    TextSemanticsCorpusforMappings,
)
from deeponto.onto.graph import IterativeMappingExtension
from deeponto.onto import Ontology
from deeponto.onto.mapping import OntoMappings
from deeponto.evaluation.align_eval import pred_thresholding
from deeponto.utils import detect_path, create_path, uniqify
from deeponto.utils.logging import banner_msg
from deeponto import SavedObj
from . import OntoAlignBase


class BERTMap(OntoAlignBase):
    def __init__(
        self,
        src_onto: Ontology,
        tgt_onto: Ontology,
        tokenizer: Tokenizer,
        bert_args: BertArguments,  # arguments for BERT fine-tuning
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
            rel="=",
            n_best=n_best,
            is_trainable=True,
            is_val_model_select=True,
            # if validation mappings are not provided, we use the default hyperparams
            default_hyperparams={"threshold": 0.9995, "map_type": "src2tgt"},
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

        # Refinement initialization
        # NOTE: init of global_match_dir is moved on OntoAlign Class
        # self.global_match_dir = self.saved_path + "/global_match"
        self.global_match_refined_dir = self.global_match_dir + "/refined"
        self.map_extender = None
        # if validation mappings are not provided, we use the default hyperparams
        # self.best_hyperparams = {"threshold": 0.999, "map_type": "src2tgt"}

    ##################################################################################
    ###                            Corpora & Fine-tune                             ###
    ##################################################################################

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
            bert_classifier = BertTrainerForSequenceClassification(
                bert_args=self.bert_args,
                train_data=self.fine_tune_data["train"],
                val_data=self.fine_tune_data["val"],
                test_data=self.fine_tune_data["test"],
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
        self.bert_args.bert_checkpoint = f"{self.fine_tune_model_path}/checkpoint-{best_checkpoint}"
        self.bert_classifier = BertStaticForSequenceClassification(self.bert_args)

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
        self.main_corpora = SavedObj.load_json(self.corpora_path + "/train-val.json")
        banner_msg("Corpora Statistics (Train-Val)")
        SavedObj.print_json(self.main_corpora["stats"])
        if detect_path(self.corpora_path + "/val.maps.json"):
            self.val_maps_corpus = SavedObj.load_json(self.corpora_path + "/val.maps.json")
            banner_msg("Corpora Statistics (Val-Maps)")
            SavedObj.print_json(self.val_maps_corpus["stats"])
        if detect_path(self.corpora_path + "/test.maps.json"):
            self.test_maps_corpus = SavedObj.load_json(self.corpora_path + "/test.maps.json")
            banner_msg("Corpora Statistics (Test-Maps)")
            SavedObj.print_json(self.test_maps_corpus["stats"])

    def load_fine_tune_data(self, split_ratio: float = 0.1):
        """Get data for fine-tuning from the corpora
        """
        banner_msg("Fine-tuning Data")
        if not detect_path(self.fine_tune_data_path):
            fine_tune_data = dict()
            fine_tune_data["stats"] = dict()
            print(
                f"Splitting main corpora into training and validation ({split_ratio * 100}%) data ..."
            )
            main_data = self.main_corpora["positives"] + self.main_corpora["negatives"]
            main_train, main_val = train_test_split(main_data, test_size=split_ratio)
            fine_tune_data["train"] = main_train
            fine_tune_data["val"] = main_val
            fine_tune_data["test"] = []
            if self.val_maps_corpus:
                print("Get additional validation data from validation mappings ...")
                # TODO: we do not care about duplicates here because label pairs from mappings are of higher importance
                fine_tune_data["val"] += (
                    self.val_maps_corpus["positives"] + self.val_maps_corpus["negatives"]
                )
            if self.test_maps_corpus:
                print("Get additional testing data from testing mappings ...")
                print(
                    "\t=> These testing mapppings do not make any decision on model selection ..."
                )
                fine_tune_data["test"] += (
                    self.test_maps_corpus["positives"] + self.test_maps_corpus["negatives"]
                )
            fine_tune_data["stats"]["n_train"] = len(fine_tune_data["train"])
            fine_tune_data["stats"]["n_val"] = len(fine_tune_data["val"])
            fine_tune_data["stats"]["n_test"] = len(fine_tune_data["test"])
            create_path(self.fine_tune_data_path)
            SavedObj.save_json(fine_tune_data, self.fine_tune_data_path + "/data.json")
        else:
            print(
                "found an existing fine-tune data directory, delete it and re-run if it's empty ..."
            )
        self.fine_tune_data = SavedObj.load_json(self.fine_tune_data_path + "/data.json")
        SavedObj.print_json(self.fine_tune_data["stats"])

    ##################################################################################
    ###                            Mapping Refinement                              ###
    ##################################################################################

    def refinement(self, map_type_to_extend: str, extension_threshold: float):
        """Apply mapping refinement as a post-processing for the scored mappings
        """

        ############################## Mapping Extensions #############################
        banner_msg(f"Mapping Refinement: Extension ({map_type_to_extend})")

        # we extend on both src2tgt and tgt2src if combined is the best
        logmap_lines = []
        map_file_path_to_refine = ""

        if map_type_to_extend == "src2tgt" or map_type_to_extend == "combined":
            src2tgt_maps = OntoMappings.from_saved(self.global_match_dir + "/src2tgt")
            self.mapping_extension(src2tgt_maps, "src2tgt", extension_threshold)
            src2tgt_maps_extended = OntoMappings.from_saved(
                self.global_match_refined_dir + "/src2tgt.extended"
            )
            # for logmap repair formatting
            logmap_lines += BERTMap.repair_formatting(
                src2tgt_maps_extended,
                self.best_hyperparams["threshold"],
                self.global_match_refined_dir + f"/src2tgt.extended",
            )
            map_file_path_to_refine = (
                self.global_match_refined_dir + f"/src2tgt.extended/src2tgt.logmap.txt"
            )

        if map_type_to_extend == "tgt2src" or map_type_to_extend == "combined":
            tgt2src_maps = OntoMappings.from_saved(self.global_match_dir + "/tgt2src")
            self.mapping_extension(tgt2src_maps, "tgt2src", extension_threshold)
            tgt2src_maps_extended = OntoMappings.from_saved(
                self.global_match_refined_dir + "/tgt2src.extended"
            )
            # for logmap repair formatting
            logmap_lines += BERTMap.repair_formatting(
                tgt2src_maps_extended,
                self.best_hyperparams["threshold"],
                self.global_match_refined_dir + f"/tgt2src.extended",
            )
            map_file_path_to_refine = (
                self.global_match_refined_dir + f"/tgt2src.extended/tgt2src.logmap.txt"
            )

        # for logmap repair formatting when type=combined; we merge src2tgt and tgt2src
        if map_type_to_extend == "combined":
            logmap_lines = uniqify(logmap_lines)
            with open(self.global_match_refined_dir + "/combined.logmap.txt", "w+") as f:
                f.writelines(logmap_lines)
            map_file_path_to_refine = self.global_match_refined_dir + f"/combined.logmap.txt"

        ############################## Mapping Repair #############################
        banner_msg(f"Mapping Refinement: Repair ({map_type_to_extend})")
        self.mapping_repair(
            formatted_file_path=map_file_path_to_refine,
            output_dir=self.global_match_refined_dir,
            map_type_to_repair=map_type_to_extend,
        )

    def mapping_extension(
        self, selected_maps: OntoMappings, map_type_to_extend: str, extension_threshold: float
    ):
        """Apply the iterative mapping extension algorithm
        """
        if detect_path(self.global_match_refined_dir + f"/{map_type_to_extend}.extended"):
            print(
                f"found existing extended {map_type_to_extend} mappings; skip mapping extension ..."
            )
            return
        self.map_extender = IterativeMappingExtension(
            self.src_onto,
            self.tgt_onto,
            selected_maps,
            self.ent_pair_score,
            extension_threshold,
            self.logger,
        )
        self.map_extender.run_extension(max_iter=10)
        self.map_extender.onto_mappings.save_instance(
            self.global_match_refined_dir + f"/{map_type_to_extend}.extended"
        )
        self.logger.info("Mapping Extension Finished\n")

    @staticmethod
    def repair_formatting(onto_mappings: OntoMappings, best_val_threshold: float, output_dir: str):
        """Formatting the mappings into LogMap format
        """
        preds = pred_thresholding(onto_mappings, best_val_threshold)
        lines = []
        for src_ent_iri, tgt_ent_iri in preds:
            if onto_mappings.flag == "src2tgt":
                score = onto_mappings.map_dict[src_ent_iri][tgt_ent_iri]
            elif onto_mappings.flag == "tgt2src":
                score = onto_mappings.map_dict[tgt_ent_iri][src_ent_iri]
            else:
                raise ValueError(f"Unknown mapping flag: {onto_mappings.flag}")
            lines.append(f"{src_ent_iri}|{tgt_ent_iri}|=|{score}|CLS\n")
        formatted_file = output_dir + f"/{onto_mappings.flag}.logmap.txt"
        with open(formatted_file, "w+") as f:
            f.writelines(lines)
        return lines

    def mapping_repair(self, formatted_file_path: str, output_dir: str, map_type_to_repair: str):
        """Apply java commands of LogMap DEBUGGER
        """
        if detect_path(self.global_match_refined_dir + f"/{map_type_to_repair}.repaired"):
            print(f"found existing extended {map_type_to_repair} mappings; skip mapping repair ...")
            return
        # apply java commands of LogMap DEBUGGER
        repair_tool_dir = deeponto.__file__.replace("__init__.py", "models/align/logmap_java")
        src_onto_path = os.path.abspath(self.src_onto.owl_path)
        tgt_onto_path = os.path.abspath(self.tgt_onto.owl_path)
        repair_saved_path = os.path.abspath(f"{output_dir}/{map_type_to_repair}.repaired")
        formatted_file_path = os.path.abspath(formatted_file_path)
        create_path(repair_saved_path)
        repair_command = (
            f"java -jar {repair_tool_dir}/logmap-matcher-4.0.jar DEBUGGER "
            + f"file:{src_onto_path} file:{tgt_onto_path} TXT {formatted_file_path}"
            + f" {repair_saved_path} false true"
        )
        self.logger.info(f"Run the following java command for repair:\n{repair_command}")
        repair_process = subprocess.Popen(repair_command.split(" "))
        try:
            _, _ = repair_process.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            repair_process.kill()
            _, _ = repair_process.communicate()

        repaired_file = (
            f"{output_dir}/{map_type_to_repair}.repaired/mappings_repaired_with_LogMap.tsv"
        )
        with open(repaired_file, "r") as f:
            lines = f.readlines()
        formatted_repaired_file = (
            f"{output_dir}/{map_type_to_repair}.repaired/{map_type_to_repair}.maps.tsv"
        )
        with open(formatted_repaired_file, "w+") as f:
            f.write("SrcEntity\tTgtEntity\tScore\n")
            for line in lines:
                src_ent_iri, tgt_ent_iri, score = line.split("\t")
                f.write(f"{src_ent_iri}\t{tgt_ent_iri}\t{score}")

        # after repair the direction is always src2tgt
        repaired_mappings = OntoMappings.read_table_mappings(
            formatted_repaired_file, flag="final_output" 
        )
        repaired_mappings.save_instance(repair_saved_path)
        # Save another copy of output mappings
        repaired_mappings.save_instance(self.global_match_dir)
        self.logger.info("Mapping Repair Finished\n")

    ##################################################################################
    ###                            Mapping Computation                             ###
    ##################################################################################

    def string_match(self, src_ent_labs: List[str], tgt_ent_labs: List[str]):
        """Predict `easy` mappings by applying string-matching
        """
        overlap_labs = set(src_ent_labs).intersection(set(tgt_ent_labs))
        return int(len(overlap_labs) > 0)

    def ent_pair_score(self, src_ent_iri: str, tgt_ent_iri: str):
        """Compute mapping score between a cross-ontology entity pair
        """
        src_ent_labs = self.src_onto.iri2labs[src_ent_iri]
        tgt_ent_labs = self.tgt_onto.iri2labs[tgt_ent_iri]
        # apply string-match before using bert
        if self.apply_string_match:
            prelim_score = self.string_match(src_ent_labs, tgt_ent_labs)
            if prelim_score == 1.0:
                return prelim_score
        # apply BERT classifier and define mapping score := Average(SynonymScores)
        src_tgt_lab_product = list(itertools.product(src_ent_labs, tgt_ent_labs))
        # only one element tensor is able to be extracted as a scalar by .item()
        return torch.mean(self.bert_classifier(src_tgt_lab_product)).item()

    def fixed_src_ent_pair_score(self, src_ent_iri: str, tgt_cand_iris: List[str]):
        """Compute mapping scores between a source entity and a batch of target entities
        """
        mappings_for_ent = super().fixed_src_ent_pair_score(src_ent_iri, tgt_cand_iris)
        # follow naming convention in global_match_for_ent and add dummy cand selection scores
        tgt_cands = [(t, 1.0) for t in tgt_cand_iris]
        tgt_cands_for_bert = []

        # for string_match
        if self.apply_string_match:
            src_ent_labs = self.src_onto.iri2labs[src_ent_iri]
            for tgt_cand_iri, _ in tgt_cands:
                tgt_ent_labs = self.tgt_onto.iri2labs[tgt_cand_iri]
                mapping_score = self.string_match(src_ent_labs, tgt_ent_labs)
                if mapping_score > 0:
                    # save mappings only with positive mapping scores
                    mappings_for_ent.append(
                        self.set_mapping(src_ent_iri, tgt_cand_iri, mapping_score)
                    )
                else:
                    tgt_cands_for_bert.append((tgt_cand_iri, 1.0))

        # for bert synonym classifier
        tgt_cands = tgt_cands_for_bert
        labs_batches = self.batched_lab_products_for_ent(
            src_ent_iri, tgt_cands, self.bert_args.batch_size_for_prediction
        )  # [{"labs": [], "lens": []}]
        batch_base_idx = 0  # after each batch, the base index will be increased by # of covered target candidates

        for labs_batch in labs_batches:
            batch_scores = self.bert_classifier(labs_batch["labs"])
            pooled_scores = self.batch_pooling(batch_scores, labs_batch["lens"])
            for i in range(len(pooled_scores)):
                score = pooled_scores[i]
                idx = i + batch_base_idx
                tgt_ent_iri = tgt_cands[idx][0]
                mappings_for_ent.append(self.set_mapping(src_ent_iri, tgt_ent_iri, score.item()))
            batch_base_idx += len(
                labs_batch["lens"]
            )  # num of lens is exactly the num of tgt candidates in this batch

        mappings_for_ent = mappings_for_ent.sorted()
        self.logger.info(f"[{self.flag}: {src_ent_iri}] {mappings_for_ent}\n")
        return mappings_for_ent

    def global_mappings_for_ent(self, src_ent_iri: str):
        """Compute cross-ontology mappings for a source entity
        """
        mappings_for_ent = super().global_mappings_for_ent(src_ent_iri)
        # select target candidates and compute score for each
        tgt_cands = self.idf_select_for_ent(src_ent_iri)  # [(tgt_id, idf_score)]

        # for string_match
        if self.apply_string_match:
            src_ent_labs = self.src_onto.iri2labs[src_ent_iri]
            for tgt_cand_iri, _ in tgt_cands:
                tgt_ent_labs = self.tgt_onto.iri2labs[tgt_cand_iri]
                mapping_score = self.string_match(src_ent_labs, tgt_ent_labs)
                if mapping_score > 0:
                    # save mappings only with positive mapping scores
                    mappings_for_ent.append(
                        self.set_mapping(src_ent_iri, tgt_cand_iri, mapping_score)
                    )
                    # output only the top (k=n_best) scored mappings
            # return the mappings if there are any string match results
            if len(mappings_for_ent) > 0:
                n_best_mappings_for_ent = mappings_for_ent.topk(self.n_best)
                self.logger.info(f"[{self.flag}: {src_ent_iri}] {n_best_mappings_for_ent}\n")
                return n_best_mappings_for_ent

        # for bert synonym classifier
        labs_batches = self.batched_lab_products_for_ent(
            src_ent_iri, tgt_cands, self.bert_args.batch_size_for_prediction
        )  # [{"labs": [], "lens": []}]
        batch_base_idx = 0  # after each batch, the base index will be increased by # of covered target candidates
        n_best_scores = torch.tensor([-1] * self.n_best).to(self.bert_classifier.device)
        n_best_idxs = torch.tensor([-1] * self.n_best).to(self.bert_classifier.device)

        for labs_batch in labs_batches:
            batch_scores = self.bert_classifier(labs_batch["labs"])
            pooled_scores = self.batch_pooling(batch_scores, labs_batch["lens"])
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
                labs_batch["lens"]
            )  # num of lens is exactly the num of tgt candidates in this batch

        for idx, score in zip(n_best_idxs, n_best_scores):
            # ignore too small values or intial values (-1.0) for dummy mappings
            if score.item() >= 0.0:
                tgt_ent_iri = tgt_cands[idx.item()][0]
                mappings_for_ent.append(self.set_mapping(src_ent_iri, tgt_ent_iri, score.item()))

        # add a dummy mapping for this entity if no mappings found
        if not mappings_for_ent:
            mappings_for_ent.append(self.set_mapping(src_ent_iri, "NullEntity", 0.0))

        # output only the top (k=n_best) scored mappings
        n_best_mappings_for_ent = mappings_for_ent.topk(self.n_best)
        self.logger.info(f"[{self.flag}: {src_ent_iri}] {n_best_mappings_for_ent}\n")
        return n_best_mappings_for_ent

    def batch_pooling(self, batch_scores: torch.Tensor, batch_lens: List[int]) -> torch.Tensor:
        """Split the tensors by specified lengths and compute the mean for each part
        """
        seq_of_scores = torch.split(batch_scores, split_size_or_sections=batch_lens)
        pooled_batch_scores = [torch.mean(chunk) for chunk in seq_of_scores]
        return torch.stack(pooled_batch_scores)
