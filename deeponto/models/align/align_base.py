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
"""Class for ontology alignment, requiring:

1. Mapping computation for a cross-ontology entity pair;
2. Full alignment computation w/o selection heuristic.

"""

from itertools import chain
from typing import List, Tuple, Optional, Iterable
from multiprocessing_on_dill import Process, Manager
import numpy as np
import os

from deeponto.onto import Ontology
from deeponto.onto.mapping import *
from deeponto.onto.text import Tokenizer, text_utils
from deeponto.utils.logging import create_logger, banner_msg
from deeponto.utils import detect_path
from deeponto.evaluation.align_eval import global_match_select
from deeponto import FlaggedObj
from deeponto.utils import create_path


class OntoAlignBase(FlaggedObj):
    def __init__(
        self,
        src_onto: Ontology,
        tgt_onto: Ontology,
        tokenizer: Tokenizer,
        cand_pool_size: Optional[int] = 200,
        rel: str = "=",
        n_best: Optional[int] = 10,
        is_trainable: bool = False,
        is_val_model_select: bool = False,
        default_hyperparams: Optional[dict] = None,
        saved_path: str = "",
    ):

        super().__init__(src_onto, tgt_onto)

        self.tokenizer = tokenizer
        self.cand_pool_size = cand_pool_size
        self.rel = rel
        self.saved_path = os.path.abspath(saved_path)  # absolute path is needed for java repair
        self.set_mapping = lambda src_ent_name, tgt_ent_name, mapping_score: EntityMapping(
            src_ent_name, tgt_ent_name, self.rel, mapping_score
        )
        self.new_mapping_list = lambda: EntityMappingList()
        self.logger = create_logger(f"{type(self).__name__}", saved_path=self.saved_path)
        self.n_best = n_best
        self.is_trainable = is_trainable
        self.is_val_model_select = is_val_model_select

        self.src2tgt_mappings = self.load_mappings("src2tgt", "global_match")
        self.tgt2src_mappings = self.load_mappings("tgt2src", "global_match")

        # for hyperparam/model selection
        self.global_match_dir = self.saved_path + "/global_match"
        # if validation mappings are not provided, we use the default hyperparams
        self.best_hyperparams = default_hyperparams

    ##################################################################################
    ###                        compute entity pair mappings                        ###
    ##################################################################################

    def pair_score(self, tbc_mappings: OntoMappings, flag: str):
        """Compute mappings for intput src-tgt entity pairs
        """
        # change side according to given
        while not self.flag == flag:
            self.switch()
        self.logger.info(
            f'Pair-score and rank input "{self.rel}" Mappings: {self.src_onto.owl.name} ==> {self.tgt_onto.owl.name}\n'
        )
        prefix = self.flag.split("2")[1]  # src or tgt
        # maximum number of mappings is the number of opposite ontology classes
        max_num_mappings = len(getattr(self, f"{prefix}_onto").classes)
        # temp = self.n_best
        self.n_best = max_num_mappings  # change n_best to all possible mappings
        mappings = self.load_mappings(flag, "pair_score")
        # self.n_best = temp
        i = 0
        for src_ent_iri, tgt2score in tbc_mappings.map_dict.items():
            if src_ent_iri in mappings.map_dict.keys():
                self.logger.info(f"skip prediction for {src_ent_iri} as already computed ...")
                continue
            # src_ent_id = self.src_onto.class2idx[src_ent_iri]
            # tgt_cand_ids = [self.tgt_onto.class2idx[t] for t in tgt2score.keys()]
            pred_maps = self.fixed_src_ent_pair_score(src_ent_iri, list(tgt2score.keys()))
            mappings.add_many(*pred_maps)
            # intermediate saving for every 100 entities
            if i % 100 == 0:
                mappings.save_instance(f"{self.saved_path}/pair_score/{self.flag}")
            i += 1
        self.logger.info("Task Finished\n")
        mappings.save_instance(f"{self.saved_path}/pair_score/{self.flag}")

    def fixed_src_ent_pair_score(self, src_ent_iri: str, tgt_cand_iris: List[str]):
        """Compute mapping scores between a source entity and a batch of target entities
        """
        banner_msg(f"Compute Mappings for Entity: " + \
            f"{self.src_onto.name_from_iri(src_ent_iri)} ({self.flag})")
        mappings_for_ent = self.new_mapping_list()
        # TODO: followed by individual implementations
        return mappings_for_ent

    def ent_pair_score(self, src_ent_iri: str, tgt_ent_iri: str) -> float:
        """Compute mapping score between a cross-ontology entity pair
        """
        raise NotImplementedError

    ##################################################################################
    ###                        compute global mappings                             ###
    ##################################################################################

    def global_match(
        self,
        num_procs: Optional[int] = None,
        match_src2tgt: bool = True,
        match_tgt2src: bool = True,
    ):
        """Compute alignment for both src2tgt and tgt2src
        """
        self.renew()
        if match_src2tgt:
            self.global_mappings_for_onto_multi_procs(
                num_procs
            ) if num_procs else self.global_mappings_for_onto()
        self.switch()
        if match_tgt2src:
            self.global_mappings_for_onto_multi_procs(
                num_procs
            ) if num_procs else self.global_mappings_for_onto()
        self.renew()

    def current_global_mappings(self):
        return getattr(self, f"{self.flag}_mappings")

    def load_mappings(self, flag: str, mode: str):
        """Create a new OntoMappings or load from saved one if any ...
        """
        flag_mappings = OntoMappings(flag=flag, n_best=self.n_best, rel=self.rel)
        saved_mappigs_path = f"{self.saved_path}/{mode}/{flag}"
        if detect_path(saved_mappigs_path):
            # raise ValueError(saved_mappigs_path)
            flag_mappings = OntoMappings.from_saved(saved_mappigs_path)
            print(f"found existing {flag} mappings, skip predictions for the saved classes ...")
        return flag_mappings

    def global_mappings_for_onto_multi_procs(self, num_procs: int):
        """Compute mappings for all entities in the current source ontology but distributed
        to multiple processes
        """
        # manager for collecting mappings from different procs
        manager = Manager()
        return_dict = manager.dict()
        # suggested by huggingface when doing multi-threading
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        def async_compute(proc_idx: int, return_dict: dict, src_ent_iri_chunk: Iterable[str]):
            return_dict[proc_idx] = self.global_mappings_for_ent_chunk(
                src_ent_iri_chunk, intermediate_saving=False
            )

        self.logger.info(
            f'Compute "{self.rel}" Mappings: {self.src_onto.owl.name} ==> {self.tgt_onto.owl.name}\n'
        )

        # split entity ids into {num_procs} chunks
        src_ent_iri_chunks = np.array_split(list(self.src_onto.iri2labs.keys()), num_procs)

        # start proc for each chunk
        jobs = []
        for i in range(num_procs):
            p = Process(target=async_compute, args=(i, return_dict, src_ent_iri_chunks[i]))
            jobs.append(p)
            p.start()

        # block the main thread until all procs finished
        for p in jobs:
            p.join()

        # save the output mappings
        mappings = self.current_global_mappings()
        for ent_mappings in return_dict.values():
            mappings.add_many(*ent_mappings)
        self.logger.info("Task Finished\n")
        mappings.save_instance(f"{self.saved_path}/global_match/{self.flag}")

    def global_mappings_for_onto(self):
        """Compute mappings for all entities in the current source ontology
        """
        self.logger.info(
            f'Compute "{self.rel}" Mappings: {self.src_onto.owl.name} ==> {self.tgt_onto.owl.name}\n'
        )
        # save the output mappings
        self.global_mappings_for_ent_chunk(self.src_onto.iri2labs.keys())
        self.logger.info("Task Finished\n")
        # saving the last batch
        mappings = self.current_global_mappings()
        mappings.save_instance(f"{self.saved_path}/global_match/{self.flag}")

    def global_mappings_for_ent_chunk(
        self,
        src_ent_iri_chunk: Iterable[str],
        save_step: int = 100,
        intermediate_saving: bool = True,
    ):
        """Compute cross-ontology mappings for a chunk of source entities,
        Note: save time especially for evaluating on Hits@K, MRR, etc.
        """
        mappings = self.current_global_mappings()
        mappings_for_chunk = []
        i = 0
        for src_ent_iri in src_ent_iri_chunk:
            if src_ent_iri in mappings.map_dict.keys():
                self.logger.info(f"skip prediction for {src_ent_iri} as already computed ...")
                continue
            cur_mappings = self.global_mappings_for_ent(src_ent_iri)
            mappings.add_many(*cur_mappings)
            mappings_for_chunk += cur_mappings
            if intermediate_saving and i % save_step == 0:
                mappings.save_instance(f"{self.saved_path}/global_match/{self.flag}")
                self.logger.info("Save currently computed mappings ...")
            i += 1
        return mappings_for_chunk

    def global_mappings_for_ent(self, src_ent_iri: str) -> EntityMappingList:
        """Compute cross-ontology mappings for a source entity
        """
        banner_msg("Compute Mappings for Entity: " + \
            f"{self.src_onto.name_from_iri(src_ent_iri)} ({self.flag})")
        mappings_for_ent = self.new_mapping_list()
        # TODO: followed by individual implementations
        return mappings_for_ent

    def idf_select_for_ent(self, src_ent_iri: str) -> Tuple[str, float]:
        """Select candidates in target ontology for a given source entity
        """
        src_ent_labs = self.src_onto.iri2labs[src_ent_iri]
        src_ent_toks = self.tokenizer.tokenize_all(src_ent_labs)
        # TODO: could have more candidate selection methods in future
        tgt_cands = self.tgt_onto.idf_select(
            src_ent_toks, self.cand_pool_size
        )  # [(ent_id, idf_score)]
        return tgt_cands

    def hyperparams_select(
        self,
        train_ref_path: Optional[str],
        val_ref_path: Optional[str],
        test_ref_path: Optional[str],
        null_ref_path: Optional[str],
        num_procs: int = 10,
    ):
        """Do hyperparam tuning on validation set before choosing which 
        {src2tgt, tgt2src, combined} mappings to be refined
        """
        # use default mapping type for extension if no validation available
        if not val_ref_path:
            print("Validation mappings are not provided; use default mapping type: src2tgt")
            return self.best_hyperparams["map_type"]  # which by default is "src2tgt"

        # validate and choose the best mapping type for mapping extension
        val_results_dir = self.global_match_dir + "/val_results"
        if detect_path(val_results_dir + "/best_hyperparams.val.json"):
            print(
                "found an existing hyperparam results on validation set,"
                + " delete it and re-run if it's empty ..."
            )
        else:
            create_path(val_results_dir)
            global_match_select(
                self.global_match_dir,
                train_ref_path,
                val_ref_path,
                test_ref_path,
                null_ref_path,
                num_procs,
            )
        self.best_hyperparams = SavedObj.load_json(
            val_results_dir + "/best_hyperparams.val.json"
        )
        banner_msg("Best Validation Hyperparams Before Refinement")
        del self.best_hyperparams["best_f1"]
        SavedObj.print_json(self.best_hyperparams)
        return self.best_hyperparams["map_type"]

    ##################################################################################
    ###                        other auxiliary functions                           ###
    ##################################################################################

    def lab_products_for_ent(
        self, src_ent_iri: str, tgt_cands: List[Tuple[str, float]]
    ) -> Tuple[List[Tuple[str, str]], List[int]]:
        """Compute Catesian Product between a source entity's labels and its selected 
        target entities' labels, with each block length recorded
        """
        src_sents, tgt_sents = [], []
        product_lens = []
        src_ent_labs = self.src_onto.iri2labs[src_ent_iri]
        # tgt_cands = self.idf_select_for_ent(src_ent_id)
        for tgt_cand_iri, _ in tgt_cands:
            tgt_ent_labs = self.tgt_onto.iri2labs[tgt_cand_iri]
            src_out, tgt_out = text_utils.lab_product(src_ent_labs, tgt_ent_labs)
            assert len(src_out) == len(tgt_out)
            product_lens.append(len(src_out))
            src_sents += src_out
            tgt_sents += tgt_out
        return list(zip(src_sents, tgt_sents)), product_lens

    def batched_lab_products_for_ent(
        self, src_ent_iri: str, tgt_cands: List[Tuple[str, float]], batch_size: int
    ):
        """Compute the batched Catesian Product between a source entity's labels and its selected 
        target entities' labels; batches are distributed according to block lengths
        """
        lab_products, product_lens = self.lab_products_for_ent(src_ent_iri, tgt_cands)
        batches = []
        cur_batch = {"labs": [], "lens": []}
        cur_lab_pointer = 0
        for i in range(len(product_lens)):  # which is the size of candidate pool
            cur_length = product_lens[i]
            cur_labs = lab_products[cur_lab_pointer : cur_lab_pointer + cur_length]
            cur_batch["labs"] += cur_labs
            cur_batch["lens"].append(cur_length)
            # collect when the batch is full or for the last set of label pairs
            if sum(cur_batch["lens"]) > batch_size or i == len(product_lens) - 1:
                # deep copy is necessary for dictionary data
                batches.append(cur_batch)
                cur_batch = {"labs": [], "lens": []}
            cur_lab_pointer += cur_length
        # small check for the algorithm
        assert lab_products == list(chain.from_iterable([b["labs"] for b in batches]))
        assert product_lens == list(chain.from_iterable([b["lens"] for b in batches]))
        return batches
