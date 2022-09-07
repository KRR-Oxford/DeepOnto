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
from datasets import load_dataset
from openprompt.data_utils import InputExample
from .subs_pairs import SubsumptionPairGenerator


def load_prompt_data(dataset_path: str = "multi_nli", *splits: str):
    """Load a datast from huggingface datasets and transform to openprompt examples.
    """
    data_dict = load_dataset(dataset_path)
    prompt_data_dict = dict()
    splits = list(splits) if splits else data_dict.keys()
    i = 0
    for split in splits:
        x_samples = []
        for samp in data_dict[split]:
            inp = InputExample(guid=i, meta=samp, label=samp["label"])
            x_samples.append(inp)
            i += 1
        prompt_data_dict[split] = x_samples
    return prompt_data_dict