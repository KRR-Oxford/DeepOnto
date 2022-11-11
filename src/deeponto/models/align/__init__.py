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
from .align_base import OntoAlignBase
from .bertmap import BERTMap
from .string_match import StringMatch
from .edit_sim import EditSimilarity
from .align_pipeline import OntoAlignPipeline

# implemented models
learning_based_models = ["bertmap"]
rule_based_models = ["string_match", "edit_sim"]
multi_procs_models = ["string_match", "edit_sim"]
implemented_models = learning_based_models + rule_based_models

# support alignment modes
supported_modes = ["global_match", "pair_score"]
eval_modes = ["global_match", "local_rank"]
