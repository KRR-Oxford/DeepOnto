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
"""
BERTStatic class for handling BERT embeddings and pre-trained/fine-tuned BERT models in eval mode

"static" here means no gradient shift happened...
"""
from typing import Tuple, List
import torch
import pandas as pd
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score
import numpy as np
import random

from deeponto.utils import Tokenizer, FileUtils


class BERTSynonymClassifier:
    """A BERT class for BERTMap consisting of a BERT model and a downstream binary classifier for synonyms.
    
    Attributes:
        pretrained_path (str): The path to the checkpoint of a pre-trained BERT model.
        output_path (str): The path to the output BERT model (usually fine-tuned).
        max_length_for_input (int): The maximum length of an input sequence.
        num_epochs_for_training (int): The number of epochs for training a BERT model.
        batch_size_for_training (int): The batch size for training a BERT model.
        batch_size_for_prediction (int): The batch size for making predictions.
        for_training (bool): Set to `True` if the model is loaded for training.
        training_data (Dataset, optional): Data for training the model if `for_training` is set to `True`. Defaults to `None`.
        validation_data (Dataset, optional): Data for validating the model if `for_training` is set to `True`. Defaults to `None`.
        testing_data (Dataset, optional): Data for testing the model regardless of the value of `for_training`. Defaults to `None`.
        training_args (TrainingArguments, optional): Training arguments for training the model if `for_training` is set to `True`. Defaults to `None`.
        trainer (Trainer, optional): The model trainer fed with `training_args`. Defaults to `None`.
    """

    def __init__(
        self,
        pretrained_path: str,
        output_path: str,
        max_length_for_input: int,
        num_epochs_for_training: float,
        batch_size_for_training: int,
        batch_size_for_prediction: int,
        for_training: bool,
        training_data: List[Tuple[str, str, int]] = None,  # (sentence1, sentence2, label)
        validation_data: List[Tuple[str, str, int]] = None,
        testing_data: List[Tuple[str, str, int]] = None,
    ):
        # Load the pretrained BERT model from the given path
        print(f"Loading a BERT model from: {pretrained_path}.")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.args.bert_checkpoint, output_hidden_states=True if for_training else False
        )
        self.tokenizer = Tokenizer.from_pretrained(pretrained_path)

        self.output_path = output_path
        self.max_length_for_input = max_length_for_input
        self.num_epochs_for_training = num_epochs_for_training
        self.batch_size_for_training = batch_size_for_training
        self.batch_size_for_prediction = batch_size_for_prediction
        self.for_training = self.for_training
        self.training_data = None
        self.validation_data = None
        self.testing_data = None
        self.data_stat = {}
        self.training_args = None
        self.trainer = None

        # load the pre-trained BERT model and set it to eval mode (static)
        if not for_training:
            print("The BERT model is set to eval mode for making predictions.")
            self.model.eval()
            # TODO: to implement multi-gpus for inference
            self.device = self.get_device(device_num=0)
            self.model.to(self.device)
        # load the pre-trained BERT model for fine-tuning
        else:
            if not training_data:
                raise RuntimeError(
                    "Training data should be provided when `for_training` is `True`."
                )
            if not validation_data:
                raise RuntimeError(
                    "Validation data should be provided when `for_training` is `True`."
                )
            # load data (max_length is used for truncation)
            self.training_data = self.load_dataset(training_data)
            self.validation_data = self.load_dataset(validation_data)
            if testing_data:
                self.testing_data = self.load_dataset(testing_data)
            self.data_stat = {
                "num_training": len(self.training_data),
                "num_validation": len(self.validation_data),
                "num_testing": len(self.testing_data) if testing_data else None,
            }
            print(f"Data statistics:\n{FileUtils.print_dict(self.data_stat)}")

            # generate training arguments
            epoch_steps = (
                len(self.training_data) // self.batch_size_for_training
            )  # total steps of an epoch
            if torch.cuda.device_count() > 0:
                epoch_steps = (
                    epoch_steps // torch.cuda.device_count()
                )  # to deal with multi-gpus case
            # keep logging steps consisitent even for small batch size
            # report logging on every 0.02 epoch
            logging_steps = int(epoch_steps * 0.02)
            # eval on every 0.2 epoch
            eval_steps = 10 * logging_steps
            # generate the training arguments
            self.training_args = TrainingArguments(
                output_dir=self.output_path,
                num_train_epochs=self.num_epochs,
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
                load_best_model_at_end=True,
                save_total_limit=2,
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

    def train(self):
        """Start training the BERT model.
        """
        if not self.for_training:
            raise RuntimeError("Training cannot be started with `for_training` set to `False`.")
        self.trainer.train()
        
        
    def predict(self, sent_pairs: List[Tuple[str, str]]):
        r"""Run prediction pipeline for the synonym classification.
        
        Return the `softmax` probailities of predicting pairs as synonyms (`index=1`).
        """
        inputs = self.process_inputs(sent_pairs)
        with torch.no_grad():
            return self.softmax(self.model(**inputs).logits)[:, 1]
        

    def load_dataset(self, data: List[Tuple[str, str, int]]) -> Dataset:
        r"""Load the list of `(sentence1, sentence2, label)` samples into a `datasets.Dataset`.
        """
        data_df = pd.DataFrame(data, columns=["sent1", "sent2", "labels"])
        dataset = Dataset.from_pandas(data_df)
        dataset = dataset.map(
            lambda examples: self.tokenizer._tokenizer(
                examples["sent1"],
                examples["sent2"],
                max_length=self.max_length_for_input,
                truncation=True,
            ),
            batched=True,
            batch_size=1024,
            num_proc=10,
        )
        return dataset

    def process_inputs(self, sent_pairs: List[Tuple[str, str]]):
        r"""Process input sentence pairs for the BERT model.
        
        Transform the sentences into BERT input embeddings and load them into the device.
        This function is called only when the BERT model is about to make predictions (`eval` mode).
        """
        return self.tokenizer._tokenizer(
            sent_pairs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length_for_input,
            truncation=True,
        ).to(self.device)

    @staticmethod
    def compute_metrics(pred):
        """Add more evaluation metrics into the training log.
        """
        # TODO: currently only accuracy is added, will expect more in the future if needed
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    @staticmethod
    def get_device(device_num: int = 0):
        """Get a device (GPU or CPU) for the torch model
        """
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
        """Set random seed for reproducible results.
        """
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
