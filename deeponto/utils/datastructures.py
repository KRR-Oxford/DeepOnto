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

from anytree import NodeMixin, RenderTree
from typing import List, Optional


def uniqify(ls):
    """Return a list of unique elements without messing around the order"""
    non_empty_ls = list(filter(lambda x: x != "", ls))
    return list(dict.fromkeys(non_empty_ls))


def sort_dict_by_values(dic: dict, desc: bool = True, k: Optional[int] = None):
    """Return a sorted dict by values with first k reserved if provided."""
    sorted_items = list(sorted(dic.items(), key=lambda item: item[1], reverse=desc))
    return dict(sorted_items[:k])


class RangeNode(NodeMixin):
    """A tree implementation for ranges (without partial overlap).

    - parent node's range fully covers child node's range, e.g., `[1, 10]` is a parent of `[2, 5]`.
    - partial overlap between ranges not allowed, e.g., `[2, 4]` and `[3, 5]` cannot appear in the same `RangeNodeTree`.
    - non-overlap ranges are on different branches.
    - child nodes are ordered according to their relative positions.
    """

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
        r"""Modified compare function for a range.

        !!! note
        
            There are three kinds of comparisons:

            - $R_1 \leq R_2$: if range $R_1$ is completely contained in range $R_2$.
            - $R_1 \gt R_2$: if range $R_2$ is completely contained in range $R_1$.
            - `"irrelevant"`: if range $R_1$ and range $R_2$ have no overlap.

        NOTE that partial overlap is not allowed.
        """
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
        """A sorting function that sorts the nodes by their starting positions."""
        temp = {sib: sib.start for sib in nodes}
        return list(dict(sorted(temp.items(), key=lambda item: item[1])).keys())

    def insert_child(self, node: RangeNode):
        """Child nodes have a smaller (inclusive) range
        e.g., `[2, 5]` is a child of `[1, 6]`.
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
