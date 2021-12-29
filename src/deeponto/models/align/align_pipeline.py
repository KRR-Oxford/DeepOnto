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

from typing import Optional, List, Tuple

from deeponto.utils import create_path
from deeponto.onto import Ontology
from deeponto.onto.text import Tokenizer
from deeponto.models import OntoPipeline
from . import StringMatch, EditSimilarity


class OntoAlignPipeline(OntoPipeline):
    def __init__(
        self,
        model_name: str,
        saved_path: str,
        config_path: str,
        src_onto_path: str,
        tgt_onto_path: str,
    ):
        super().__init__(model_name, saved_path, config_path)
        self.paths.src_onto = self.complete_path("src_onto")
        self.paths.tgt_onto = self.complete_path("tgt_onto")

        # load tokenizer (type = pretrained or rule_based)
        tkz_load = getattr(Tokenizer, f"from_{self.config.tokenizer.type}")
        self.tokenizer = tkz_load(self.config.tokenizer.path)

        # load src and tgt ontologies
        self.src_onto = self.load_onto("src", src_onto_path)
        self.tgt_onto = self.load_onto("tgt", tgt_onto_path)

        # load align object
        self.model = self.load_model()

    def run(
        self,
        mode: str,
        ent_name_pairs: Optional[List[Tuple[str, str]]] = None,
        num_procs: Optional[int] = None,
    ):
        """Run the whole pipeline
        """
        super().run()

        # train the learning-based models
        if self.model.is_trainable:
            self.model.train(**self.config.train)

        # make prediction according mode
        if mode == "global_match":
            self.model.global_match(num_procs)
        elif mode == "pair_score":
            assert ent_name_pairs != None
            # return the mapping dict for subsequent evaluation
            return self.model.pair_score(ent_name_pairs)
        else:
            raise ValueError(f"Unknown mode: {mode}, please choose from [global_match, scoring].")

    def load_onto(self, flag: str, new_onto_path: str):
        """Load ontology from saved or new data path
        """
        saved_onto_path = getattr(self.paths, f"{flag}_onto")
        onto = self.from_saved(saved_onto_path, is_onto=True)
        # if nothing saved
        if not onto:
            onto = Ontology.from_new(new_onto_path, self.config.lab_props, self.tokenizer)
            onto.save_instance(saved_onto_path)
            print(f"Load `new` {flag} ontology from: {new_onto_path}")
        else:
            print(f"Load `saved` {flag} ontology from: {saved_onto_path}")
        return onto

    def load_model(self):
        """Load alignment model according to model name
        """
        # create directory for saving align model
        self.paths.model = self.complete_path(self.model_name)
        create_path(self.paths.model)

        # load the model according to model name
        if self.model_name == "bertmap":
            pass
        elif self.model_name == "string_match":
            align_model = StringMatch(
                src_onto=self.src_onto,
                tgt_onto=self.tgt_onto,
                tokenizer=self.tokenizer,
                cand_pool_size=self.config.search.cand_pool_size,
                n_best=self.config.search.n_best,
                saved_path=self.paths.model,
            )
        elif self.model_name == "edit_sim":
            align_model = EditSimilarity(
                src_onto=self.src_onto,
                tgt_onto=self.tgt_onto,
                tokenizer=self.tokenizer,
                cand_pool_size=self.config.search.cand_pool_size,
                n_best=self.config.search.n_best,
                saved_path=self.paths.model,
            )
        else:
            raise ValueError(f"{self.model_name} is not an implemented model.")

        return align_model
