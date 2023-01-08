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
"""Class for input configurations"""

import click

from deeponto import SavedObj
from deeponto.utils.logging import banner_msg


class InputConfig(SavedObj):
    def __init__(self):
        super().__init__("config")

    @classmethod
    def new_config(cls):
        """Set up new config template from command lines
        """
        config = dict()
        finished = False
        banner_msg(f"New {type(cls).__name__} Template")
        while not finished:
            cls.add_param_group(config)
            finished = not click.confirm("Do you want to continue?")
        cls.preview_and_save(config)

    @staticmethod
    def add_param_group(param_dict: dict):
        """Construct param group with maximum two levels
        """
        top_param = click.prompt("Please enter a valid (identifier) parameter name", type=str)
        nested = click.confirm(f'Do you want to add a nested parameter under "{top_param}]"')
        # 1-level {param: value}
        if not nested:
            param_dict[top_param] = click.prompt(f'Please enter the value for "{top_param}"')
        else:
            param_dict[top_param] = dict()
        # 2-levels {param: {sub_param: value}}
        while nested:
            sub_param = click.prompt("Please enter a valid (identifier) parameter name", type=str)
            param_dict[top_param][sub_param] = click.prompt(
                f'Please enter the value for "{sub_param}"'
            )
            nested = click.confirm(f'More nested parameters for "{top_param}"')
        banner_msg(f'add a new param group under "{top_param}"')

    @classmethod
    def preview_and_save(cls, config):
        banner_msg("Preview of the Config")
        SavedObj.print_json(config)
        saved = click.confirm("Do you want to save the config?")
        if saved:
            save_path = click.prompt("Please enter the saved file path")
            cls.save_json(config, save_path)

    @classmethod
    def load_config(cls, config_json_path):
        """Load saved config
        """
        return dict(cls.load_json(config_json_path))

    @classmethod
    def edit_config(cls, config_json_path):
        """Edit existing config through command lines
        """
        config = cls.load_json(config_json_path)
        banner_msg(f"Choose Config Params to Edit")
        top_params = list(config.keys())
        finished = False
        while not finished:
            for i in range(len(top_params)):
                name = top_params[i]
                print(f'[{i}]: "{name}"')
            selected = top_params[click.prompt("Please choose a number", type=int)]
            top_val = config[selected]
            if isinstance(top_val, dict):
                sub_params = list(top_val.keys())
                for i in range(len(sub_params)):
                    name = sub_params[i]
                    print(f'[{i}]: "{name}"')
                sub_selected = sub_params[
                    click.prompt(
                        f'Please choose a number for the nested parameters under "{selected}"',
                        type=int,
                    )
                ]
                print(f'Existed value for "{sub_selected}": {config[selected][sub_selected]}')
                config[selected][sub_selected] = click.prompt(f'New value for "{sub_selected}"')
            else:
                print(f'Existed value for "{selected}": {config[selected]}')
                config[selected] = click.prompt(f'New value for "{selected}"')
            finished = not click.confirm("Do you want to continue?")
        cls.preview_and_save(config)
