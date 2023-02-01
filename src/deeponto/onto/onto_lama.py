# Credit to: https://github.com/thunlp/OpenPrompt/blob/main/experiments/cli.py
"""Script for running openprompt models for OntoLAMA."""

import os
from typing import List, Callable, Optional
import random
import logging
from collections import defaultdict
from datasets import load_dataset

from openprompt.trainer import ClassificationRunner, GenerationRunner
from openprompt.lm_bff_trainer import LMBFFClassificationRunner
from openprompt.protoverb_trainer import ProtoVerbClassificationRunner
from openprompt.pipeline_base import PromptForClassification, PromptForGeneration
from openprompt.utils.reproduciblity import set_seed
from openprompt.prompts import (
    load_template,
    load_verbalizer,
)
from openprompt.data_utils import FewShotSampler
from openprompt.utils.logging import config_experiment_dir, init_logger
from openprompt.config import get_config, save_config_to_yaml
from openprompt.plms import load_plm_from_config
from openprompt import PromptDataLoader
from openprompt.prompt_base import Template
from openprompt.plms.utils import TokenizerWrapper
from transformers.tokenization_utils import PreTrainedTokenizer
from yacs.config import CfgNode

from openprompt.data_utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor
from openprompt.utils.logging import logger

from deeponto.utils import FileUtils

CUR_TEMPLATE = None
CUR_VERBALIZER = None


# @credit("openprompt", "https://github.com/thunlp/OpenPrompt/blob/main/experiments/cli.py")
def build_dataloader(
    dataset: List,
    template: Template,
    tokenizer: PreTrainedTokenizer,
    tokenizer_wrapper_class: TokenizerWrapper,
    config: CfgNode,
    split: str,
):
    dataloader = PromptDataLoader(
        dataset=dataset,
        template=template,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=tokenizer_wrapper_class,
        batch_size=config[split].batch_size,
        shuffle=config[split].shuffle_data,
        teacher_forcing=config[split].teacher_forcing if hasattr(config[split], "teacher_forcing") else None,
        predict_eos_token=True if config.task == "generation" else False,
        **config.dataloader,
    )
    example = template.incorporate_text_example(random.choice(dataset))
    logger = logging.getLogger()
    logger.info(f"transformed example: {example}")
    return dataloader


# @credit("openprompt", "https://github.com/thunlp/OpenPrompt/blob/main/experiments/cli.py")
def run_openprompt(config, args):
    """Main entry for running the OpenPrompt script.

    Modified from "https://github.com/thunlp/OpenPrompt/blob/main/experiments/cli.py".
    """
    global CUR_TEMPLATE, CUR_VERBALIZER
    # exit()
    # init logger, create log dir and set log level, etc.
    if args.resume and args.test:
        raise Exception("cannot use flag --resume and --test together")
    if args.resume or args.test:
        config.logging.path = EXP_PATH = args.resume or args.test
    else:
        EXP_PATH = config_experiment_dir(config)
        init_logger(
            os.path.join(EXP_PATH, "log.txt"),
            config.logging.file_level,
            config.logging.console_level,
        )
        # save config to the logger directory
        save_config_to_yaml(config)

    # load dataset. The valid_dataset can be None
    train_dataset, valid_dataset, test_dataset, Processor = InferenceDataProcessor.load_inference_dataset(
        config, test=args.test is not None or config.learning_setting == "zero_shot"
    )

    # main
    if config.learning_setting == "full":
        res = trainer(
            EXP_PATH,
            config,
            Processor,
            resume=args.resume,
            test=args.test,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
        )
    elif config.learning_setting == "few_shot":
        if config.few_shot.few_shot_sampling is None:
            raise ValueError("use few_shot setting but config.few_shot.few_shot_sampling is not specified")
        seeds = config.sampling_from_train.seed
        res = 0
        for seed in seeds:
            if not args.test:
                sampler = FewShotSampler(
                    num_examples_per_label=config.sampling_from_train.num_examples_per_label,
                    also_sample_dev=config.sampling_from_train.also_sample_dev,
                    num_examples_per_label_dev=config.sampling_from_train.num_examples_per_label_dev,
                )
                train_sampled_dataset, valid_sampled_dataset = sampler(
                    train_dataset=train_dataset, valid_dataset=valid_dataset, seed=seed
                )
                result = trainer(
                    os.path.join(EXP_PATH, f"seed-{seed}"),
                    config,
                    Processor,
                    resume=args.resume,
                    test=args.test,
                    train_dataset=train_sampled_dataset,
                    valid_dataset=valid_sampled_dataset,
                    test_dataset=test_dataset,
                )
            else:
                result = trainer(
                    os.path.join(EXP_PATH, f"seed-{seed}"),
                    config,
                    Processor,
                    test=args.test,
                    test_dataset=test_dataset,
                )
            res += result
        res /= len(seeds)
    elif config.learning_setting == "zero_shot":
        res = trainer(
            EXP_PATH,
            config,
            Processor,
            zero=True,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
        )

    return config, CUR_TEMPLATE, CUR_VERBALIZER


def trainer(
    EXP_PATH,
    config,
    Processor,
    train_dataset=None,
    valid_dataset=None,
    test_dataset=None,
    resume=None,
    test=None,
    zero=False,
):
    global CUR_TEMPLATE, CUR_VERBALIZER

    if not os.path.exists(EXP_PATH):
        os.mkdir(EXP_PATH)
    config.logging.path = EXP_PATH
    # set seed
    set_seed(config.reproduce.seed)

    # load the pretrained models, its model, tokenizer, and config.
    plm_model, plm_tokenizer, plm_config, plm_wrapper_class = load_plm_from_config(config)

    # define template and verbalizer
    if config.task == "classification":
        # define prompt
        template = load_template(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config)
        verbalizer = load_verbalizer(
            config=config,
            model=plm_model,
            tokenizer=plm_tokenizer,
            plm_config=plm_config,
            classes=Processor.labels,
        )
        # load prompt’s pipeline model
        prompt_model = PromptForClassification(
            plm_model, template, verbalizer, freeze_plm=config.plm.optimize.freeze_para
        )

    elif config.task == "generation":
        template = load_template(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config)
        prompt_model = PromptForGeneration(
            plm_model,
            template,
            freeze_plm=config.plm.optimize.freeze_para,
            gen_config=config.generation,
        )
    else:
        raise NotImplementedError(
            f"config.task {config.task} is not implemented yet. Only classification and generation are supported."
        )

    # process data and get data_loader
    train_dataloader = (
        build_dataloader(train_dataset, template, plm_tokenizer, plm_wrapper_class, config, "train")
        if train_dataset
        else None
    )
    valid_dataloader = (
        build_dataloader(valid_dataset, template, plm_tokenizer, plm_wrapper_class, config, "dev")
        if valid_dataset
        else None
    )
    test_dataloader = (
        build_dataloader(test_dataset, template, plm_tokenizer, plm_wrapper_class, config, "test")
        if test_dataset
        else None
    )

    if config.task == "classification":
        if config.classification.auto_t or config.classification.auto_v:
            runner = LMBFFClassificationRunner(
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                test_dataset=test_dataset,
                template=template,
                verbalizer=verbalizer,
                config=config,
            )
        elif config.verbalizer == "proto_verbalizer":
            runner = ProtoVerbClassificationRunner(
                model=prompt_model,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                test_dataloader=test_dataloader,
                id2label=Processor.id2label,
                config=config,
            )
        else:
            runner = ClassificationRunner(
                model=prompt_model,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                test_dataloader=test_dataloader,
                id2label=Processor.id2label,
                config=config,
            )
    elif config.task == "generation":
        runner = GenerationRunner(
            model=prompt_model,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            test_dataloader=test_dataloader,
            config=config,
        )

    CUR_TEMPLATE = template
    CUR_VERBALIZER = verbalizer
    logger = logging.getLogger()
    logger.info(f"Label classes: {verbalizer.classes}")
    logger.info(f"Label words: {verbalizer.label_words}")
    if zero:
        res = runner.test()
    elif test:
        res = runner.test(ckpt="best")
    elif resume:
        res = runner.run(ckpt="last")
    else:
        res = runner.run()
    return res


class InferenceDataProcessor(DataProcessor):
    """Class for processing inference data.

    Modified according to "https://github.com/thunlp/OpenPrompt/blob/main/experiments/cli.py".
    """

    def __init__(self, apply_binary: bool):
        super().__init__()
        self.labels = ["entailment", "neutral", "contradiction"]
        self.label2idx = {"entailment": 0, "neutral": 1, "contradiction": 2}
        if apply_binary:
            self.labels = ["entailment", "contradiction"]
            self.label2idx = {"entailment": 0, "contradiction": 1}

    def get_examples(self, data_dir, split):
        data_path = os.path.join(data_dir, "{}.tsv".format(split))
        examples = self.load_prompt_data_from_table(data_path, self.label2idx)
        return examples

    @staticmethod
    def load_prompt_data_from_table(
        tabular_data_path: str,
        label2idx: dict,
        premise_pattern: Optional[Callable] = lambda x: x,  # extra pattern for premise, default is no change
        hypothesis_pattern: Optional[Callable] = lambda x: x,  # extra pattern for hypothesis, default is no change
    ):
        """Load an inference dataset containing (premise-hypothesis) pairs
        from the .json file.
        """
        df = FileUtils.read_table(tabular_data_path)
        prompt_data = []
        stats = defaultdict(lambda: 0)
        for i, dp in df.iterrows():
            samp = {
                "premise": premise_pattern(dp["Premise"]),
                "hypothesis": hypothesis_pattern(dp["Hypothesis"]),
                "label": dp["Label"],
            }
            label_idx = label2idx[dp["Label"]]
            inp = InputExample(guid=i, meta=samp, label=label_idx)
            prompt_data.append(inp)
            stats[dp["Label"]] += 1
        print("Load inference dataset with the following statistics:")
        FileUtils.print_dict(stats)
        return prompt_data

    @classmethod
    def load_inference_dataset(cls, config: CfgNode, return_class=True, test=False):
        r"""A plm loader using a global config.
        It will load the train, valid, and test set (if exists) simulatenously.

        Args:
            config (:obj:`CfgNode`): The global config from the CfgNode.
            return_class (:obj:`bool`): Whether return the data processor class
                        for future usage.

        Returns:
            :obj:`Optional[List[InputExample]]`: The train dataset.
            :obj:`Optional[List[InputExample]]`: The valid dataset.
            :obj:`Optional[List[InputExample]]`: The test dataset.
            :obj:"
        """
        dataset_config = config.dataset

        processor = cls(True)

        train_dataset = None
        valid_dataset = None
        if not test:
            try:
                train_dataset = processor.get_train_examples(dataset_config.path)
            except FileNotFoundError:
                logger.warning(f"Has no training dataset in {dataset_config.path}.")
            try:
                valid_dataset = processor.get_dev_examples(dataset_config.path)
            except FileNotFoundError:
                logger.warning(f"Has no validation dataset in {dataset_config.path}.")

        test_dataset = None
        try:
            test_dataset = processor.get_test_examples(dataset_config.path)
        except FileNotFoundError:
            logger.warning(f"Has no test dataset in {dataset_config.path}.")
        # checking whether donwloaded.
        if (train_dataset is None) and (valid_dataset is None) and (test_dataset is None):
            logger.error(
                "Dataset is empty. Either there is no download or the path is wrong. "
                + "If not downloaded, please `cd datasets/` and `bash download_xxx.sh`"
            )
            exit()
        if return_class:
            return train_dataset, valid_dataset, test_dataset, processor
        else:
            return train_dataset, valid_dataset, test_dataset

    @staticmethod
    def load_prompt_data_from_huggingface(dataset_path: str = "multi_nli", *splits: str):
        """Load a datast from huggingface datasets and transform to openprompt examples."""
        data_dict = load_dataset(dataset_path)
        prompt_data_dict = dict()
        splits = list(splits) if splits else data_dict.keys()
        i = 0
        for split in splits:
            x_samples = []
            for samp in data_dict[split]:
                inp = InputExample(guid=i, meta=samp, label=samp["label"])
                x_samples.append(inp)
                i += 1
            prompt_data_dict[split] = x_samples
        return prompt_data_dict
