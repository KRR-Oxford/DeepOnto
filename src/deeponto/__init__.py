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

# the following code is credited to the mOWL library 
import jpype
import os
import mowl

def init_jvm(memory):
    dirname = os.path.dirname(mowl.__file__)
    jars_dir = os.path.join(dirname, "lib/")
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
