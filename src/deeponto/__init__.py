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

# the following code is credited to the mOWL library under the BSD 3-Clause License:
# https://github.com/bio-ontology-research-group/mowl/blob/main/LICENSE
import jpype
import jpype.imports  # very important for basic Java dependencies!
import os
import platform

def init_jvm(memory):
    jars_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "lib/")
    # jars_dir = os.path.join(os.path.dirname(os.path.realpath(mowl.__file__)), "lib/")
    
    if not os.path.exists(jars_dir):
        raise FileNotFoundError(f"JAR files not found. Make sure that the lib directory exists \
and contains the JAR dependencies.")

    if (platform.system() == 'Windows'):
        jars = f'{str.join(";", [jars_dir + name for name in os.listdir(jars_dir)])}'
    else:
        jars = f'{str.join(":", [jars_dir + name for name in os.listdir(jars_dir)])}'
        
    if not jpype.isJVMStarted():

        jpype.startJVM(
            jpype.getDefaultJVMPath(), "-ea",
            f"-Xmx{memory}",
            "-Djava.class.path=" + jars,
            convertStrings=False)
        
    if jpype.isJVMStarted():
        print(f"{memory} maximum memory allocated to JVM.")
        print("JVM started successfully.")
