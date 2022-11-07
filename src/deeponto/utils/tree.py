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
"""A ternary tree implementation for integer ranges (without partial overlap)."""

from __future__ import annotations

class RangeNode:
    def __init__(self, start: int, end: int, **kwargs):
        assert start <= end
        self.start = start
        self.end = end
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.left = None
        self.right = None
        self.container = None
        self.content = None
    
    def insert(self, start: int, end: int, **kwargs):
        assert start <= end
        if end < self.start:  # inserted node is on the left
            if self.left:
                self.left.insert(start, end, **kwargs)
            else:
                self.left = RangeNode(start, end, **kwargs)
        elif self.end < start:  # inserted node is on the right
            if self.right:
                self.right.insert(start, end, **kwargs)
            else:
                self.right = RangeNode(start, end, **kwargs)
        # the following two parts are reversible edges
        elif self.start <= start and end <= self.end:  # inserted node is smaller 
            new_node = RangeNode(start, end, **kwargs)
            if self.content:
                if start <= self.content.start and self.content.end <= end:  # inserted node is in between 
                     # insert the node in between 
                     self.content.container = new_node
                     new_node.content = self.content
                     self.content = new_node
                     new_node.container = self
                else:
                    # going further down
                    self.content.insert(start, end, **kwargs)
            else:
                # just set reversible relationships
                self.content = new_node
                new_node.container = self
        elif start <= self.start and self.end <= end: # inserted node is larger
            new_node = RangeNode(start, end, **kwargs)
            # check if left and right need to be changed because of the new container
            if self.right and new_node.end < self.right.start:
                new_node.right = self.right
                self.right = None
            if self.left and self.left.end < new_node.start:
                new_node.left = self.left
            # update container
            if self.container:
                if self.container.start <= start and end <= self.container.end: # inserted node is in between
                    # insert the node in between
                    self.container.content = new_node
                    new_node.container = self.container
                    self.container = new_node
                    new_node.content = self
                else:
                    # going further up
                    self.container.insert(start, end, **kwargs)
            else:
                self.container = new_node
                new_node.content = self
        else:
            raise RuntimeError("Cannot insert a new node with partial overlap")
        
    
    def __str__(self):
        # only present downwards (down, left, right)
        printed = f"[{self.start}, {self.end}]"
        if self.content:
            printed = f"[{self.start}, {str(self.content)}, {self.end}]"
        if self.left:
            printed = f"{str(self.left)}, {printed}"
        if self.right:
            printed = f"{printed}, {str(self.right)}"
        return printed
    
    def print_root(self):
        cur_node = self
        while cur_node.container:
            cur_node = cur_node.container
        print(cur_node)
