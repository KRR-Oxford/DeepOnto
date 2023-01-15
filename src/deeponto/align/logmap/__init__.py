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
from deeponto.utils import FileUtils
import os


def run_logmap_repair(
    src_onto_path: str, tgt_onto_path: str, mapping_file_path: str, output_path: str
):
    """Run the repair module of LogMap with `java -jar`."""
    
    # find logmap directory
    logmap_path = os.path.dirname(__file__)
    
    # obtain absolute paths
    src_onto_path = os.path.abspath(src_onto_path)
    tgt_onto_path = os.path.abspath(tgt_onto_path)
    mapping_file_path = os.path.abspath(mapping_file_path)
    output_path = os.path.abspath(output_path)
    
    # run jar command
    print(f"Run the repair module of LogMap from {logmap_path}.")
    repair_command = (
        f"java -jar {logmap_path}/logmap-matcher-4.0.jar DEBUGGER "
        + f"file:{src_onto_path} file:{tgt_onto_path} TXT {mapping_file_path}"
        + f" {output_path} false true"
    )
    print(f"The jar command is:\n{repair_command}.")
    FileUtils.run_jar(repair_command)
