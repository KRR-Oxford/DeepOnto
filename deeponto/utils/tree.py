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
"""A tree implementation for ranges (without partial overlap).
- parent node's range fully covers child node's range, e.g., [1, 10] isParentOf [2, 5]
- partial overlap between ranges not allowed, e.g., [2, 4] and [3, 5] cannot appear in the same RangeNodeTree
- non-overlap ranges are on different branches 
- child nodes are ordered according to their relative positions
"""

from __future__ import annotations
from anytree import NodeMixin, RenderTree
from typing import List
import math

class RangeNode(NodeMixin):
    def __init__(self, start, end, **kwargs):
        if start >= end:
            raise RuntimeError("invalid start and end positions ...")
        self.start = start
        self.end = end
        for k, v in kwargs.items():
            setattr(self, k, v)
        super().__init__()
        
    # def __eq__(self, other: RangeNode):
    #     return self.start == other.start and self.end == other.end
        
    def __gt__(self, other: RangeNode):
        if other.start <= self.start and self.end <= other.end:
            return False
        elif self.start <= other.start and other.end <= self.end:
            return True
        elif other.end < self.start or self.end < other.start:
            # print("compared ranges are irrelevant ...")
            return "irrelevant"
        else:
            raise RuntimeError("compared ranges have partial overlap ...")
        
    @staticmethod
    def sort_by_start(nodes: List[RangeNode]):
        """A sorting function that sorts the nodes by their starting positions
        """
        temp = {
            sib: sib.start for sib in nodes
        }
        return list(dict(sorted(temp.items(), key=lambda item: item[1])).keys())
    
    def insert_child(self, node: RangeNode):
        """Child nodes have a smaller (inclusive) range
        e.g., [2, 5] is a child of [1, 6]
        """
        if node > self:
            raise RuntimeError("invalid child node")
        if node.start == self.start and node.end == self.end:
            # duplicated node
            return
        # print(self.children)
        if self.children:
            inserted = False
            for ch in self.children:
                if (node < ch) is True:
                    # print("further down")
                    ch.insert_child(node)
                    inserted = True
                    break
                elif (node > ch) is True:
                    # print("insert in between")
                    ch.parent = node
                    # NOTE: should not break here as it could be parent of multiple children !
                    # break
            if not inserted:
                self.children = list(self.children) + [node]
                self.children = self.sort_by_start(self.children)
        else:
            node.parent = self
            self.children = [node]
    
    def __repr__(self):
        # only present downwards (down, left, right)
        printed = f"[{self.start}, {self.end}]"
        if self.children:
            printed = f"[{self.start}, {str(list(self.children))[1:-1]}, {self.end}]"
        return printed
    
    def print_tree(self):
        print(RenderTree(self))

