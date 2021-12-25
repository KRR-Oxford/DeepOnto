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
"""Define evaluation metrics for different tasks"""

from typing import Optional, Iterable

# def precision(pred: Iterable, ref: Iterable):
#     """
#     % of predictions are correct:
#         P = TP / (TP + FP)
#     """
#     if ignored:
#         num_pre_ignored = len(pred) - len(self.pre_unignored_set)
#         self.num_ignored += num_pre_ignored
#     # True Positive = the number of unignored prediction mappings that are True
#     tp = len(self.pre_unignored_set.intersection(self.ref_set))  
#     # False Positive = the number of unignored prediction mappings that are False
#     fp = len(self.pre_set) - tp - num_pre_ignored  
#     return tp / (tp + fp)

# class OntoEvaluator:

#     na_vals = bertmap.na_vals
#     namespaces = bertmap.namespaces

#     def __init__(
#         self,
#         pre_tsv: Union[str, DataFrame],
#         ref_tsv: Union[str, DataFrame],
#         ref_ignored_tsv: Optional[Union[str, DataFrame]] = None,
#         threshold: float = 0.0,
#     ):

#         # filter the prediction mappings according to similarity scores
#         self.pre = self.read_mappings(pre_tsv, threshold=threshold)
#         self.id_maps = []  # mapping of same iris should be ignored
#         self.identity_mapping_check()
#         self.num_ignored = len(self.id_maps)

#         # reference mappings and illegal mappings to be ignored
#         self.ref = self.read_mappings(ref_tsv)
#         if ref_ignored_tsv is None:
#             self.ref_ignored = None
#         else:
#             self.ref_ignored = self.read_mappings(ref_ignored_tsv)
        
#         # convert list to set for fast computation
#         self.pre_set = set(self.pre)
#         self.ref_set = set(self.ref)
#         if self.ref_ignored:
#             self.ref_ignored_set = set(self.ref_ignored)
#             self.pre_unignored_set = self.pre_set - self.ref_ignored_set
#             self.ref_unignored_set = self.ref_set - self.ref_ignored_set

#         # compute Precision, Recall and Macro-F1
#         try:
#             self.P = self.precision()
#             self.R = self.recall()
#             self.F1 = self.f1()
#         except:
#             self.P = -1.0
#             self.R = -1.0
#             self.F1 = -1.0

#     def precision(self) -> float:
#         """
#         % of predictions are correct:
#             P = TP / (TP + FP)
#         """
#         if self.ref_ignored:
#             num_pre_ignored = len(self.pre_set) - len(self.pre_unignored_set)
#             self.num_ignored += num_pre_ignored
#         # True Positive = the number of unignored prediction mappings that are True
#         tp = len(self.pre_unignored_set.intersection(self.ref_set))  
#         # False Positive = the number of unignored prediction mappings that are False
#         fp = len(self.pre_set) - tp - num_pre_ignored  
#         return tp / (tp + fp)

#     def recall(self) -> float:
#         """
#         % of correct retrieved
#             R = TP / (TP + FN)
#         """
#         if self.ref_ignored:
#             num_ref_ignored = len(self.ref_set) - len(self.ref_unignored_set)
#             self.num_ignored += num_ref_ignored
#         # True Positive = the number of unignored reference mappings that are Positive
#         tp = len(self.ref_unignored_set.intersection(self.pre_set))
#         # False Negative = the number of unignored reference mappings that are Negative
#         fn = len(self.ref_set) - tp - num_ref_ignored
#         return tp / (tp + fn)

#     def f1(self) -> float:
#         return 2 * self.P * self.R / (self.P + self.R)
    
#     def identity_mapping_check(self) -> None:
#         for p_map in self.pre:
#             if p_map.split("\t")[0] == p_map.split("\t")[1]:
#                 self.id_maps.append(p_map)
#         self.pre = list(set(self.pre) - set(self.id_maps))
#         if len(self.id_maps) > 0:
#             print(f"detect and delete {len(self.id_maps)} mappings of identical iris ...")
