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
"""Class for running implemented models."""

from typing import Optional
import torch

from deeponto.utils import create_path
from deeponto.onto import Ontology
from deeponto.onto.mapping import OntoMappings
from deeponto.onto.text import Tokenizer
from deeponto.models import OntoPipelineBase
from deeponto.bert import BertArguments
from . import StringMatch, EditSimilarity, BERTMap


class OntoAlignPipeline(OntoPipelineBase):
    def __init__(
        self,
        model_name: str,
        saved_path: str,
        config_path: str,
        src_onto_path: str,
        tgt_onto_path: str,
    ):
        super().__init__(model_name, saved_path, config_path)
        self.paths["src_onto"] = self.complete_path("src_onto")
        self.paths["tgt_onto"] = self.complete_path("tgt_onto")

        # load tokenizer (type = pretrained or rule_based)
        tkz_type = self.config["tokenizer"]["type"]
        tkz_path = self.config["tokenizer"]["path"]
        tkz_load = getattr(Tokenizer, f"from_{tkz_type}")
        self.tokenizer = tkz_load(tkz_path)

        # load src and tgt ontologies
        self.src_onto = self.load_onto("src", src_onto_path)
        self.tgt_onto = self.load_onto("tgt", tgt_onto_path)

        # load align object
        self.model = self.load_model()

    def run(
        self,
        mode: str,
        tbc_mappings: Optional[OntoMappings] = None,
        tbc_flag: Optional[str] = None,
        num_procs: Optional[int] = None,
    ):
        """Run the whole pipeline
        """
        super().run()

        # train the learning-based models
        if self.model.is_trainable:
            self.model.train()
            torch.cuda.empty_cache()

        # make prediction according mode
        if mode == "global_match":
            self.model.global_match(
                num_procs=num_procs,
                match_src2tgt=self.config["search"]["match_src2tgt"],
                match_tgt2src=self.config["search"]["match_tgt2src"],
            )
            map_type = self.model.hyperparams_select(
                self.config["corpora"]["train_mappings_path"],
                self.config["corpora"]["val_mappings_path"],
                self.config["corpora"]["test_mappings_path"],
                self.config["corpora"]["null_mappings_path"],
            )
            torch.cuda.empty_cache()

            # mapping refinement for bertmap
            if self.model_name == "bertmap":
                self.model.refinement(map_type, self.config["search"]["extension_threshold"])
                torch.cuda.empty_cache()

        elif mode == "pair_score":
            assert tbc_mappings != None
            self.model.pair_score(tbc_mappings, tbc_flag)
            torch.cuda.empty_cache()

        else:
            raise ValueError(f"Unknown mode: {mode}, please choose from [global_match, scoring].")

    def load_onto(self, flag: str, new_onto_path: str):
        """Load ontology from saved or new data path
        """
        saved_onto_path = self.paths[f"{flag}_onto"]
        onto = self.from_saved(saved_onto_path, is_onto=True)
        # if nothing saved
        if not onto:
            onto = Ontology.from_new(new_onto_path, self.config["lab_props"], self.tokenizer)
            onto.save_instance(saved_onto_path)
            # onto.destroy_owl_cache()
            onto = Ontology.from_saved(saved_onto_path)
            print(f"Load `new` {flag} ontology from: {new_onto_path}")
        else:
            print(f"Load `saved` {flag} ontology from: {saved_onto_path}")
        return onto

    def load_model(self):
        """Load alignment model according to model name
        """
        # create directory for saving align model
        self.paths["model"] = self.complete_path(self.model_name)
        create_path(self.paths["model"])

        # load the model according to model name
        if self.model_name == "bertmap":
            # get arguments for BERT
            self.paths["bert"] = self.complete_path("bertmap/fine_tune/model")
            self.bert_args = BertArguments(
                bert_checkpoint=self.config["bert"]["pretrained_path"],
                output_dir=self.paths["bert"],
                num_epochs=float(self.config["bert"]["num_epochs"]),
                batch_size_for_training=self.config["bert"]["batch_size_for_training"],
                batch_size_for_prediction=self.config["bert"]["batch_size_for_prediction"],
                max_length=self.config["bert"]["max_length"],
                device_num=self.config["bert"]["device_num"],
                early_stop_patience=self.config["bert"]["early_stop_patience"],
                resume_from_ckp=self.config["bert"]["resume_from_ckp"],
            )

            # load mappings if any
            ref_mappings = {"train": None, "val": None, "test": None}
            if self.config["corpora"]["train_mappings_path"]:
                ref_mappings["train"] = OntoMappings.read_table_mappings(
                    self.config["corpora"]["train_mappings_path"]
                )
            if self.config["corpora"]["val_mappings_path"]:
                ref_mappings["val"] = OntoMappings.read_table_mappings(
                    self.config["corpora"]["val_mappings_path"]
                )
            if self.config["corpora"]["test_mappings_path"]:
                ref_mappings["test"] = OntoMappings.read_table_mappings(
                    self.config["corpora"]["test_mappings_path"]
                )

            # load auxiliary ontologies if any
            aux_ontos = []
            aux_count = 0
            for aux_onto_path in self.config["corpora"]["aux_onto_paths"]:
                aux_flag = f"aux_{aux_count}"
                self.paths[f"{aux_flag}_onto"] = self.complete_path(f"{aux_flag}_onto")
                # no need to load tokenizer
                aux_ontos.append(self.load_onto(f"{aux_flag}", aux_onto_path))
                aux_count += 1

            # build the BERTMap model
            align_model = BERTMap(
                src_onto=self.src_onto,
                tgt_onto=self.tgt_onto,
                tokenizer=self.tokenizer,
                bert_args=self.bert_args,
                cand_pool_size=self.config["search"]["cand_pool_size"],
                n_best=self.config["search"]["n_best"],
                saved_path=self.paths["model"],
                train_mappings=ref_mappings["train"],
                validation_mappings=ref_mappings["val"],
                test_mappings=ref_mappings["test"],
                aux_ontos=aux_ontos,
                apply_transitivity=self.config["corpora"]["apply_transitivity"],
                neg_ratio=self.config["corpora"]["neg_ratio"],
                apply_string_match=self.config["search"]["apply_string_match"],
            )

        elif self.model_name == "string_match":
            align_model = StringMatch(
                src_onto=self.src_onto,
                tgt_onto=self.tgt_onto,
                tokenizer=self.tokenizer,
                cand_pool_size=self.config["search"]["cand_pool_size"],
                n_best=self.config["search"]["n_best"],
                saved_path=self.paths["model"],
            )

        elif self.model_name == "edit_sim":
            align_model = EditSimilarity(
                src_onto=self.src_onto,
                tgt_onto=self.tgt_onto,
                tokenizer=self.tokenizer,
                cand_pool_size=self.config["search"]["cand_pool_size"],
                n_best=self.config["search"]["n_best"],
                saved_path=self.paths["model"],
            )

        else:
            raise ValueError(f"{self.model_name} is not an implemented model.")

        return align_model
