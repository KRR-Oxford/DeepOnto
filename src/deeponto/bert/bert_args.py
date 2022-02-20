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
"""Class for input arguments of a BERT model"""

# import transformers
from pyats.datastructures import AttrDict

class BERTArgs:
    
    def __init__(self, 
                 bert_checkpoint: str,
                 # bert_class: str,
                 n_epochs: float,
                 batch_size: int,
                 max_length: int,
                 device_num: int,
                 ):
        
        # basic arguments
        self.bert_checkpoint = bert_checkpoint
        # huggingface class models
        # self.bert_classes = [x for x in transformers.__all__ if "AutoModel" in x]
        # if not bert_class in self.bert_classes:
        #     raise ValueError(f"{bert_class} is not one of the implemented BERT (transformers) class.")
        
        # training arguments
        self.train = AttrDict()
        self.train.n_epochs = n_epochs
        self.train.batch_size = batch_size
        self.train.max_length = max_length
        self.train.device_num = device_num
