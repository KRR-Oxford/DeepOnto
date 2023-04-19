# Copyright 2021 Yuan He. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from yacs.config import CfgNode
from datasets import load_dataset
from openprompt.data_utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor
from openprompt.utils.logging import logger


class OntoLAMADataProcessor(DataProcessor):
    """Class for processing the OntoLAMA data points."""

    def __init__(self):
        super().__init__()
        self.labels = ["negative", "positive"]

    @staticmethod
    def load_dataset(task_name: str, split: str):
        """Load a specific OntoLAMA dataset from huggingface dataset hub."""
        # TODO: remove use_auth_token after going to public
        return load_dataset("krr-oxford/OntoLAMA", task_name, split=split, use_auth_token=True)

    def get_examples(self, task_name, split):
        """Load a specific OntoLAMA dataset and transform the data points into
        input examples for prompt-based inference.
        """

        dataset = self.load_dataset(task_name, split)

        premise_name = "v_sub_concept"
        hypothesis_name = "v_super_concept"
        # different data fields for the bimnli dataset
        if "bimnli" in task_name:
            premise_name = "premise"
            hypothesis_name = "hypothesis"

        prompt_samples = []
        for samp in dataset:
            inp = InputExample(text_a=samp[premise_name], text_b=samp[hypothesis_name], label=samp["label"])
            prompt_samples.append(inp)

        return prompt_samples

    @classmethod
    def load_inference_dataset(cls, config: CfgNode, return_class=True, test=False):
        r"""A plm loader using a global config.
        It will load the train, valid, and test set (if exists) simulatenously.

        Args:
            config (CfgNode): The global config from the CfgNode.
            return_class (bool): Whether return the data processor class for future usage.

        Returns:
            (Optional[List[InputExample]]): The train dataset.
            (Optional[List[InputExample]]): The valid dataset.
            (Optional[List[InputExample]]): The test dataset.
            (Optional[OntoLAMADataProcessor]): The data processor object.
        """
        dataset_config = config.dataset

        processor = cls()

        train_dataset = None
        valid_dataset = None
        if not test:
            try:
                train_dataset = processor.get_examples(dataset_config.task_name, "train")
            except FileNotFoundError:
                logger.warning(f"Has no training dataset in krr-oxford/OntoLAMA/{dataset_config.task_name}.")
            try:
                valid_dataset = processor.get_examples(dataset_config.task_name, "validation")
            except FileNotFoundError:
                logger.warning(f"Has no validation dataset in krr-oxford/OntoLAMA/{dataset_config.task_name}.")

        test_dataset = None
        try:
            test_dataset = processor.get_examples(dataset_config.task_name, "test")
        except FileNotFoundError:
            logger.warning(f"Has no test dataset in krr-oxford/OntoLAMA/{dataset_config.task_name}.")
        # checking whether donwloaded.
        if (train_dataset is None) and (valid_dataset is None) and (test_dataset is None):
            logger.error(
                "Dataset is empty. Either there is no download or the path is wrong. "
                + "If not downloaded, please `cd datasets/` and `bash download_xxx.sh`"
            )
            exit()
        if return_class:
            return train_dataset, valid_dataset, test_dataset, processor
        else:
            return train_dataset, valid_dataset, test_dataset
