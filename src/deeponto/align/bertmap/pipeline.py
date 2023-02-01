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
import enlighten
# import transformers

from deeponto.align.mapping import ReferenceMapping
from deeponto.onto import Ontology
from deeponto.utils.decorators import paper
from deeponto.utils import FileUtils, Tokenizer
from deeponto.utils.logging import create_logger
from .text_semantics import TextSemanticsCorpora
from .bert_classifier import BERTSynonymClassifier
from .mapping_prediction import MappingPredictor
from .mapping_refinement import MappingRefiner


MODEL_OPTIONS = {"bertmap": {"trainable": True}, "bertmaplt": {"trainable": False}}
DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "default_config.yaml")
# transformers.logging.set_verbosity_info()


# @paper(
#     "BERTMap: A BERT-based Ontology Alignment System (AAAI-2022)",
#     "https://ojs.aaai.org/index.php/AAAI/article/view/20510",
# )
class BERTMapPipeline:
    r"""Class for the whole ontology alignment pipeline of $\textsf{BERTMap}$ and $\textsf{BERTMapLt}$ models.

    !!! note

        Parameters related to BERT training are `None` by default. They will be constructed for
        $\textsf{BERTMap}$ and stay as `None` for $\textsf{BERTMapLt}$.

    Attributes:
        config (CfgNode): The configuration for BERTMap or BERTMapLt.
        name (str): The name of the model, either `bertmap` or `bertmaplt`.
        output_path (str): The path to the output directory.
        src_onto (Ontology): The source ontology to be matched.
        tgt_onto (Ontology): The target ontology to be matched.
        annotation_property_iris (List[str]): The annotation property IRIs used for extracting synonyms and nonsynonyms.
        src_annotation_index (dict): A dictionary that stores the `(class_iri, class_annotations)` pairs from `src_onto` according to `annotation_property_iris`.
        tgt_annotation_index (dict): A dictionary that stores the `(class_iri, class_annotations)` pairs from `tgt_onto` according to `annotation_property_iris`.
        known_mappings (List[ReferenceMapping], optional): List of known mappings for constructing the **cross-ontology corpus**.
        auxliary_ontos (List[Ontology], optional): List of auxiliary ontolgoies for constructing any **auxiliary corpus**.
        corpora (dict, optional): A dictionary that stores the `summary` of built text semantics corpora and the sampled `synonyms` and `nonsynonyms`.
        finetune_data (dict, optional): A dictionary that stores the `training` and `validation` splits of samples from `corpora`.
        bert (BERTSynonymClassifier, optional): A BERT model for synonym classification and mapping prediction.
        best_checkpoint (str, optional): The path to the best BERT checkpoint which will be loaded after training.
        mapping_predictor (MappingPredictor): The predictor function based on class annotations, used for **global matching** or **mapping scoring**.

    """

    def __init__(self, src_onto: Ontology, tgt_onto: Ontology, config: CfgNode):
        """Initialize the BERTMap or BERTMapLt model.

        Args:
            src_onto (Ontology): The source ontology for alignment.
            tgt_onto (Ontology): The target ontology for alignment.
            config (CfgNode): The configuration for BERTMap or BERTMapLt.
        """
        # load the configuration and confirm model name is valid
        self.config = config
        self.name = self.config.model
        if not self.name in MODEL_OPTIONS.keys():
            raise RuntimeError(f"`model` {self.name} in the config file is not one of the supported.")

        # create the output directory, e.g., experiments/bertmap
        self.config.output_path = "." if not self.config.output_path else self.config.output_path
        self.config.output_path = os.path.abspath(self.config.output_path)
        self.output_path = os.path.join(self.config.output_path, self.name)
        FileUtils.create_path(self.output_path)

        # create logger and progress manager (hidden attribute) 
        self.logger = create_logger(self.name, self.output_path)
        self.enlighten_manager = enlighten.get_manager()

        # ontology
        self.src_onto = src_onto
        self.tgt_onto = tgt_onto
        self.annotation_property_iris = self.config.annotation_property_iris
        self.logger.info(f"Load the following configurations:\n{FileUtils.print_dict(self.config)}")
        config_path = os.path.join(self.output_path, "config.yaml")
        self.logger.info(f"Save the configuration file at {config_path}.")
        self.save_bertmap_config(self.config, config_path)

        # build the annotation thesaurus
        self.src_annotation_index, _ = self.src_onto.build_annotation_index(self.annotation_property_iris)
        self.tgt_annotation_index, _ = self.tgt_onto.build_annotation_index(self.annotation_property_iris)

        # provided mappings if any
        self.known_mappings = self.config.known_mappings
        if self.known_mappings:
            self.known_mappings = ReferenceMapping.read_table_mappings(self.known_mappings)

        # auxiliary ontologies if any
        self.auxiliary_ontos = self.config.auxiliary_ontos
        if self.auxiliary_ontos:
            self.auxiliary_ontos = [Ontology(ao) for ao in self.auxiliary_ontos]

        self.data_path = os.path.join(self.output_path, "data")
        # load or construct the corpora
        self.corpora_path = os.path.join(self.data_path, "text-semantics.corpora.json")
        self.corpora = self.load_text_semantics_corpora()

        # load or construct fine-tune data
        self.finetune_data_path = os.path.join(self.data_path, "fine-tune.data.json")
        self.finetune_data = self.load_finetune_data()

        # load the bert model and train
        self.bert_config = self.config.bert
        self.bert_pretrained_path = self.bert_config.pretrained_path
        self.bert_finetuned_path = os.path.join(self.output_path, "bert")
        self.bert_resume_training = self.bert_config.resume_training
        self.bert_synonym_classifier = None
        self.best_checkpoint = None
        if self.name == "bertmap":
            self.bert_synonym_classifier = self.load_bert_synonym_classifier()
            # train if the loaded classifier is not in eval mode
            if self.bert_synonym_classifier.eval_mode == False:
                self.logger.info(
                    f"Data statistics:\n \
                    {FileUtils.print_dict(self.bert_synonym_classifier.data_stat)}"
                )
                self.bert_synonym_classifier.train(self.bert_resume_training)
                # turn on eval mode after training
                self.bert_synonym_classifier.eval()
            # NOTE potential redundancy here: after training, load the best checkpoint
            self.best_checkpoint = self.load_best_checkpoint()
            if not self.best_checkpoint:
                raise RuntimeError(f"No best checkpoint found for the BERT synonym classifier model.")
            self.logger.info(f"Fine-tuning finished, found best checkpoint at {self.best_checkpoint}.")
        else:
            self.logger.info(f"No training needed; skip BERT fine-tuning.")
            
        # pretty progress bar tracking
        self.enlighten_status = self.enlighten_manager.status_bar(
            status_format=u'Global Matching{fill}Stage: {demo}{fill}{elapsed}',
            color='bold_underline_bright_white_on_lightslategray',
            justify=enlighten.Justify.CENTER, demo='Initializing',
            autorefresh=True, min_delta=0.5
        )

        # mapping predictions
        self.global_matching_config = self.config.global_matching
        self.mapping_predictor = MappingPredictor(
            output_path=self.output_path,
            tokenizer_path=self.bert_config.pretrained_path,
            src_annotation_index=self.src_annotation_index,
            tgt_annotation_index=self.tgt_annotation_index,
            bert_synonym_classifier=self.bert_synonym_classifier,
            num_raw_candidates=self.global_matching_config.num_raw_candidates,
            num_best_predictions=self.global_matching_config.num_best_predictions,
            batch_size_for_prediction=self.bert_config.batch_size_for_prediction,
            logger=self.logger,
            enlighten_manager=self.enlighten_manager,
            enlighten_status=self.enlighten_status
        )
        self.mapping_refiner = None

        # if global matching is disabled (potentially used for class pair scoring)
        if self.config.global_matching.enabled:
            self.mapping_predictor.mapping_prediction()  # mapping prediction
            if self.name == "bertmap":
                self.mapping_refiner = MappingRefiner(
                    output_path=self.output_path,
                    src_onto=self.src_onto,
                    tgt_onto=self.tgt_onto,
                    mapping_predictor=self.mapping_predictor,
                    mapping_extension_threshold=self.global_matching_config.mapping_extension_threshold,
                    mapping_filtered_threshold=self.global_matching_config.mapping_filtered_threshold,
                    logger=self.logger,
                    enlighten_manager=self.enlighten_manager,
                    enlighten_status=self.enlighten_status
                )
                self.mapping_refiner.mapping_extension()  # mapping extension
                self.mapping_refiner.mapping_repair()  # mapping repair
            self.enlighten_status.update(demo="Finished")  
        else:
            self.enlighten_status.update(demo="Skipped")  
              
        self.enlighten_status.close()

        # class pair scoring is invoked outside

    def load_or_construct(self, data_file: str, data_name: str, construct_func: Callable, *args, **kwargs):
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

        See [`TextSemanticsCorpora`][deeponto.align.bertmap.text_semantics.TextSemanticsCorpora].
        """
        data_name = "text semantics corpora"

        if self.name == "bertmap":

            def construct():
                corpora = TextSemanticsCorpora(
                    src_onto=self.src_onto,
                    tgt_onto=self.tgt_onto,
                    annotation_property_iris=self.annotation_property_iris,
                    class_mappings=self.known_mappings,
                    auxiliary_ontos=self.auxiliary_ontos,
                )
                self.logger.info(str(corpora))
                corpora.save(self.data_path)

            return self.load_or_construct(self.corpora_path, data_name, construct)

        self.logger.info(f"No training needed; skip the construction of {data_name}.")
        return None

    def load_finetune_data(self):
        r"""Load or construct fine-tuning data from text semantics corpora.

        Steps of constructing fine-tuning data from text semantics:

        1. Mix synonym and nonsynonym data.
        2. Randomly sample 90% as training samples and 10% as validation.
        """
        data_name = "fine-tuning data"

        if self.name == "bertmap":

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

    def load_bert_synonym_classifier(self):
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
            checkpoint = self.bert_pretrained_path
            eval_mode = False  # since it is for training now

        return BERTSynonymClassifier(
            loaded_path=checkpoint,
            output_path=self.bert_finetuned_path,
            eval_mode=eval_mode,
            max_length_for_input=self.bert_config.max_length_for_input,
            num_epochs_for_training=self.bert_config.num_epochs_for_training,
            batch_size_for_training=self.bert_config.batch_size_for_training,
            batch_size_for_prediction=self.bert_config.batch_size_for_prediction,
            training_data=self.finetune_data["training"],
            validation_data=self.finetune_data["validation"],
        )

    def load_best_checkpoint(self) -> Optional[str]:
        """Find the best checkpoint by searching for trainer states in each checkpoint file."""
        best_checkpoint = -1

        if os.path.exists(self.bert_finetuned_path):
            for file in os.listdir(self.bert_finetuned_path):
                # load trainer states from each checkpoint file
                if file.startswith("checkpoint"):
                    trainer_state = FileUtils.load_file(
                        os.path.join(self.bert_finetuned_path, file, "trainer_state.json")
                    )
                    checkpoint = int(trainer_state["best_model_checkpoint"].split("/")[-1].split("-")[-1])
                    # find the latest best checkpoint
                    if checkpoint > best_checkpoint:
                        best_checkpoint = checkpoint

        if best_checkpoint == -1:
            best_checkpoint = None
        else:
            best_checkpoint = os.path.join(self.bert_finetuned_path, f"checkpoint-{best_checkpoint}")

        return best_checkpoint
    
    @staticmethod
    def load_bertmap_config(config_file: Optional[str] = None):
        """Load the BERTMap configuration in `.yaml`. If the file
        is not provided, use the default configuration.
        """
        if not config_file:
            config_file = DEFAULT_CONFIG_FILE
            print(f"Use the default configuration at {DEFAULT_CONFIG_FILE}.")  
        if not config_file.endswith(".yaml"):
            raise RuntimeError("Configuration file should be in `yaml` format.")
        return CfgNode(FileUtils.load_file(config_file))

    @staticmethod
    def save_bertmap_config(config: CfgNode, config_file: str):
        """Save the BERTMap configuration in `.yaml`."""
        with open(config_file, "w") as c:
            config.dump(stream=c, sort_keys=False, default_flow_style=False)
