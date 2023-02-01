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

from typing import Tuple, List, Optional, Union
import torch
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score
import numpy as np
import random

from deeponto.utils import Tokenizer, FileUtils
from deeponto.utils.decorators import paper

# @paper(
#     "BERTMap: A BERT-based Ontology Alignment System (AAAI-2022)",
#     "https://ojs.aaai.org/index.php/AAAI/article/view/20510",
# )
class BERTSynonymClassifier:
    r"""Class for BERT synonym classifier.

    The main scoring module of $\textsf{BERTMap}$ consisting of a BERT model and a binary synonym classifier.

    Attributes:
        loaded_path (str): The path to the checkpoint of a pre-trained BERT model.
        output_path (str): The path to the output BERT model (usually fine-tuned).
        eval_mode (bool): Set to `False` if the model is loaded for training.
        max_length_for_input (int): The maximum length of an input sequence.
        num_epochs_for_training (int): The number of epochs for training a BERT model.
        batch_size_for_training (int): The batch size for training a BERT model.
        batch_size_for_prediction (int): The batch size for making predictions.
        training_data (Dataset, optional): Data for training the model if `for_training` is set to `True`. Defaults to `None`.
        validation_data (Dataset, optional): Data for validating the model if `for_training` is set to `True`. Defaults to `None`.
        training_args (TrainingArguments, optional): Training arguments for training the model if `for_training` is set to `True`. Defaults to `None`.
        trainer (Trainer, optional): The model trainer fed with `training_args` and data samples. Defaults to `None`.
        softmax (torch.nn.SoftMax, optional): The softmax layer used for normalising synonym scores. Defaults to `None`.
    """

    def __init__(
        self,
        loaded_path: str,
        output_path: str,
        eval_mode: bool,
        max_length_for_input: int,
        num_epochs_for_training: Optional[float] = None,
        batch_size_for_training: Optional[int] = None,
        batch_size_for_prediction: Optional[int] = None,
        training_data: Optional[List[Tuple[str, str, int]]] = None,  # (sentence1, sentence2, label)
        validation_data: Optional[List[Tuple[str, str, int]]] = None,
    ):
        # Load the pretrained BERT model from the given path
        self.loaded_path = loaded_path
        print(f"Loading a BERT model from: {self.loaded_path}.")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.loaded_path, output_hidden_states=eval_mode
        )
        self.tokenizer = Tokenizer.from_pretrained(loaded_path)

        self.output_path = output_path
        self.eval_mode = eval_mode
        self.max_length_for_input = max_length_for_input
        self.num_epochs_for_training = num_epochs_for_training
        self.batch_size_for_training = batch_size_for_training
        self.batch_size_for_prediction = batch_size_for_prediction
        self.training_data = None
        self.validation_data = None
        self.data_stat = {}
        self.training_args = None
        self.trainer = None
        self.softmax = None

        # load the pre-trained BERT model and set it to eval mode (static)
        if self.eval_mode:
            self.eval()
        # load the pre-trained BERT model for fine-tuning
        else:
            if not training_data:
                raise RuntimeError("Training data should be provided when `for_training` is `True`.")
            if not validation_data:
                raise RuntimeError("Validation data should be provided when `for_training` is `True`.")
            # load data (max_length is used for truncation)
            self.training_data = self.load_dataset(training_data, "training")
            self.validation_data = self.load_dataset(validation_data, "validation")
            self.data_stat = {
                "num_training": len(self.training_data),
                "num_validation": len(self.validation_data),
            }

            # generate training arguments
            epoch_steps = len(self.training_data) // self.batch_size_for_training  # total steps of an epoch
            if torch.cuda.device_count() > 0:
                epoch_steps = epoch_steps // torch.cuda.device_count()  # to deal with multi-gpus case
            # keep logging steps consisitent even for small batch size
            # report logging on every 0.02 epoch
            logging_steps = int(epoch_steps * 0.02)
            # eval on every 0.2 epoch
            eval_steps = 10 * logging_steps
            # generate the training arguments
            self.training_args = TrainingArguments(
                output_dir=self.output_path,
                num_train_epochs=self.num_epochs_for_training,
                per_device_train_batch_size=self.batch_size_for_training,
                per_device_eval_batch_size=self.batch_size_for_training,
                warmup_ratio=0.0,
                weight_decay=0.01,
                logging_steps=logging_steps,
                logging_dir=f"{self.output_path}/tensorboard",
                eval_steps=eval_steps,
                evaluation_strategy="steps",
                do_train=True,
                do_eval=True,
                save_steps=eval_steps,
                save_total_limit=2,
                load_best_model_at_end=True,
            )
            # build the trainer
            self.trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.training_data,
                eval_dataset=self.validation_data,
                compute_metrics=self.compute_metrics,
                tokenizer=self.tokenizer._tokenizer,
            )

    def train(self, resume_from_checkpoint: Optional[Union[bool, str]] = None):
        """Start training the BERT model."""
        if self.eval_mode:
            raise RuntimeError("Training cannot be started in `eval` mode.")
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    def eval(self):
        """To eval mode."""
        print("The BERT model is set to eval mode for making predictions.")
        self.model.eval()
        # TODO: to implement multi-gpus for inference
        self.device = self.get_device(device_num=0)
        self.model.to(self.device)
        self.softmax = torch.nn.Softmax(dim=1).to(self.device)

    def predict(self, sent_pairs: List[Tuple[str, str]]):
        r"""Run prediction pipeline for synonym classification.

        Return the `softmax` probailities of predicting pairs as synonyms (`index=1`).
        """
        inputs = self.process_inputs(sent_pairs)
        with torch.no_grad():
            return self.softmax(self.model(**inputs).logits)[:, 1]

    def load_dataset(self, data: List[Tuple[str, str, int]], split: str) -> Dataset:
        r"""Load the list of `(annotation1, annotation2, label)` samples into a `datasets.Dataset`."""

        def iterate():
            for sample in data:
                yield {"annotation1": sample[0], "annotation2": sample[1], "labels": sample[2]}

        dataset = Dataset.from_generator(iterate)
        # NOTE: no padding here because the Trainer class supports dynamic padding
        dataset = dataset.map(
            lambda examples: self.tokenizer._tokenizer(
                examples["annotation1"], examples["annotation2"], max_length=self.max_length_for_input, truncation=True
            ),
            batched=True,
            desc=f"Load {split} data with batch size 1000:",
        )
        return dataset

    def process_inputs(self, sent_pairs: List[Tuple[str, str]]):
        r"""Process input sentence pairs for the BERT model.

        Transform the sentences into BERT input embeddings and load them into the device.
        This function is called only when the BERT model is about to make predictions (`eval` mode).
        """
        return self.tokenizer._tokenizer(
            sent_pairs,
            return_tensors="pt",
            max_length=self.max_length_for_input,
            padding=True,
            truncation=True,
        ).to(self.device)

    @staticmethod
    def compute_metrics(pred):
        """Add more evaluation metrics into the training log."""
        # TODO: currently only accuracy is added, will expect more in the future if needed
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    @staticmethod
    def get_device(device_num: int = 0):
        """Get a device (GPU or CPU) for the torch model"""
        # If there's a GPU available...
        if torch.cuda.is_available():
            # Tell PyTorch to use the GPU.
            device = torch.device(f"cuda:{device_num}")
            print("There are %d GPU(s) available." % torch.cuda.device_count())
            print("We will use the GPU:", torch.cuda.get_device_name(device_num))
        # If not...
        else:
            print("No GPU available, using the CPU instead.")
            device = torch.device("cpu")
        return device

    @staticmethod
    def set_seed(seed_val: int = 888):
        """Set random seed for reproducible results."""
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
