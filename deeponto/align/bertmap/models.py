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

from typing import Optional, Callable
from yacs.config import CfgNode
import os
import random
# import transformers

from deeponto.align.mapping import ReferenceMapping
from deeponto.onto import Ontology
from deeponto.utils import FileUtils
from deeponto.utils.logging import create_logger
from .text_semantics import TextSemanticsCorpora
from .bert_classifier import BERTSynonymClassifier


MODEL_OPTIONS = {"bertmap": {"trainable": True}, "bertmaplt": {"trainable": False}}
DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "default_config.yaml")
# transformers.logging.set_verbosity_info()


def load_config(config_file: Optional[str] = None):
    """Load the configuration in `.yaml`. If the file
    is not provided, use the default configuration.
    """
    if not config_file:
        config_file = DEFAULT_CONFIG_FILE
        print(f"Use the default configuration at {DEFAULT_CONFIG_FILE}.")
    if not config_file.endswith(".yaml"):
        raise RuntimeError("Configuration file should be in `yaml` format.")
    return CfgNode(FileUtils.load_file(config_file))


class BERTMapPipeline:
    """Class for BERTMap and BERTMapLt models.
    
    Attributes:
        config (CfgNode): The configuration for BERTMap or BERTMapLt.
        name (str): The name of the model, either `bertmap` or `bertmaplt`.
        output_path (str): 
    
    
    """

    def __init__(self, src_onto_path: Optional[str], tgt_onto_path: Optional[str], config: CfgNode):
        """Initialize the BERTMap or BERTMapLt model.

        Args:
            src_onto_path (Optional[str]): The path to the source ontology file (overwrites `config.ontology.src_onto`)
            tgt_onto_path (Optional[str]): The path to the target ontology file (overwrittes `config.ontology.tgt_onto`)
            config (CfgNode): The configuration for BERTMap or BERTMapLt.
        """
        # load the configuration and confirm model name is valid
        self.config = config
        self.name = self.config.matching.model
        if not self.name in MODEL_OPTIONS.keys():
            raise RuntimeError(
                f"`model` {self.name} in the config file is not one of the supported."
            )
        self.trainable = MODEL_OPTIONS[self.name]["trainable"]

        # create the output directory, e.g., experiments/bertmap
        self.config.output_path = "." if not self.config.output_path else self.config.output_path
        self.config.output_path = os.path.abspath(self.config.output_path)
        self.output_path = os.path.join(self.config.output_path, self.name)
        FileUtils.create_path(self.output_path)

        # create logger
        self.logger = create_logger(self.name, self.output_path)

        # ontology
        if src_onto_path:
            self.config.ontology.src_onto = src_onto_path
        if tgt_onto_path:
            self.config.ontology.tgt_onto = tgt_onto_path
        self.src_onto = Ontology(self.config.ontology.src_onto)
        self.tgt_onto = Ontology(self.config.ontology.tgt_onto)
        self.annotation_property_iris = self.config.ontology.annotation_property_iris
        self.logger.info(f"Load configurations:\n{FileUtils.print_dict(self.config)}")
        FileUtils.save_file(dict(self.config), os.path.join(self.output_path, "config.yaml"))

        # provided mappings if any
        self.known_mappings = self.config.matching.known_mappings
        if self.known_mappings:
            self.known_mappings = ReferenceMapping.read_table_mappings(self.training_mappings)

        # auxiliary ontologies if any
        self.auxiliary_ontos = self.config.matching.auxiliary_ontos
        if self.auxiliary_ontos:
            self.auxiliary_ontos = [Ontology(ao) for ao in self.auxiliary_ontos]

        # build the annotation thesaurus
        self.src_annotation_index = self.src_onto.build_annotation_index(
            self.annotation_property_iris
        )
        self.tgt_annotation_index = self.tgt_onto.build_annotation_index(
            self.annotation_property_iris
        )

        self.data_path = os.path.join(self.output_path, "data")
        # load or construct the corpora
        self.corpora_path = os.path.join(self.data_path, "text-semantics.corpora.json")
        self.corpora = self.load_text_semantics_corpora()

        # load or construct fine-tune data
        self.finetune_data_path = os.path.join(self.data_path, "fine-tune.data.json")
        self.finetune_data = self.load_finetune_data()

        # init the bert model
        self.bert_config = self.config.matching.bert
        self.bert_pretrained_path = self.bert_config.pretrained_path
        self.bert_finetuned_path = os.path.join(self.output_path, "bert")
        self.bert_resume_training = self.bert_config.resume_training
        self.bert = self.load_bert()
        self.logger.info(f"Data statistics:\n{FileUtils.print_dict(self.bert.data_stat)}")

    def load_or_construct(
        self, data_file: str, data_name: str, construct_func: Callable, *args, **kwargs
    ):
        """Load existing data or construct a new one.
        
        An auxlirary function that checks the existence of a data file and loads it if it exists.
        Otherwise, construct new data with the input `construct_func` which is supported generate
        a local data file.
        """
        if os.path.exists(data_file):
            self.logger.info(f"Load existing {data_name} from {data_file}.")
        else:
            self.logger.info(f"Construct new {data_name} and save at {data_file}.")
            construct_func(*args, **kwargs)
        # load the data file that is supposed to be saved locally
        return FileUtils.load_file(data_file)

    def load_text_semantics_corpora(self):
        """Load or construct text semantics corpora.
        
        See [`TextSemanticsCorpora`][deeponto.align.bertmap.text_semantics_corpora.TextSemanticsCorpora].
        """
        data_name = "text semantics corpora"

        if self.trainable:

            def construct():
                corpora = TextSemanticsCorpora(
                    src_onto=self.src_onto,
                    tgt_onto=self.tgt_onto,
                    annotation_property_iris=self.annotation_property_iris,
                    class_mappings=self.known_mappings,
                    auxiliary_ontos=self.auxiliary_ontos,
                )
                corpora.save(self.data_path)

            return self.load_or_construct(self.corpora_path, data_name, construct)

        self.logger.info(f"No training needed; skip the construction of {data_name}.")
        return None

    def load_finetune_data(self):
        r"""Load or construct fine-tuning data from text semantics corpora.
        
        !!! note
        
            Steps of constructing fine-tuning data from text semantics:
                
            1. Mix synonym and nonsynonym data.
            2. Randomly sample 90% as training samples and 10% as validation.
        """
        data_name = "fine-tuning data"

        if self.trainable:

            def construct():
                finetune_data = dict()
                samples = self.corpora["synonyms"] + self.corpora["nonsynonyms"]
                random.shuffle(samples)
                split_index = int(0.9 * len(samples))  # split at 90%
                finetune_data["training"] = samples[:split_index]
                finetune_data["validation"] = samples[split_index:]
                FileUtils.save_file(finetune_data, self.finetune_data_path)
                
            return self.load_or_construct(self.finetune_data_path, data_name, construct)

        self.logger.info(f"No training needed; skip the construction of {data_name}.")
        return None

    def load_bert(self):
        """Load the BERT model from a pre-trained or a local checkpoint.
        
            - If loaded from pre-trained, it means to start training from a pre-trained model such as `bert-uncased`.
            - If loaded from local, turn on the `eval` mode for mapping predictions.
            - If `self.bert_resume_training` is `True`, it will be loaded from the latest saved checkpoint.
        """
        checkpoint = self.load_best_checkpoint()  # load the best checkpoint or nothing
        eval_mode = True
        # if no checkpoint has been found, start training from scratch OR resume training
        # no point to load the best checkpoint if resume training (will automatically search for the latest checkpoint)
        if not checkpoint or self.bert_resume_training:  
            loaded_path = self.bert_pretrained_path
            eval_mode = False  # since it is for training now
        
        return BERTSynonymClassifier(
            loaded_path=loaded_path,
            output_path=self.bert_finetuned_path,
            eval_mode=eval_mode,
            max_length_for_input=self.bert_config.max_length_for_input,
            num_epochs_for_training=self.bert_config.num_epochs_for_training,
            batch_size_for_training=self.bert_config.batch_size_for_training,
            batch_size_for_prediction=self.bert_config.batch_size_for_prediction,
            training_data=self.finetune_data["training"],
            validation_data=self.finetune_data["validation"]
        )


    def load_best_checkpoint(self) -> Optional[str]:
        """Find the best checkpoint by searching for trainer states in each checkpoint file.
        """
        best_checkpoint = -1
        
        if os.path.exists(self.bert_finetuned_path):
            for file in os.listdir(self.bert_finetuned_path):
                # load trainer states from each checkpoint file
                if file.startswith("checkpoint"):
                    trainer_state = FileUtils.load_file(
                        os.path.join(self.bert_finetuned_path, file, "trainer_state.json")
                    )
                    checkpoint = int(
                        trainer_state["best_model_checkpoint"].split("/")[-1].split("-")[-1]
                    )
                    # find the latest best checkpoint
                    if checkpoint > best_checkpoint:
                        best_checkpoint = checkpoint
                    
        if best_checkpoint == -1:
            best_checkpoint = None
        else:
            best_checkpoint = os.path.join(self.bert_finetuned_path, f"checkpoint-{best_checkpoint}")
            self.logger.info(f"Found best checkpoint at {best_checkpoint}.")
        return best_checkpoint
