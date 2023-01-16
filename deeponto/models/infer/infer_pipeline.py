# Credit to: https://github.com/thunlp/OpenPrompt/blob/main/experiments/cli.py
"""Script for running openprompt models for OntoLAMA"""

import os
import sys

main_dir = os.getcwd().split("DeepOnto")[0] + "DeepOnto/src"
sys.path.append(main_dir)

from typing import List
import random
import logging
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

from deeponto.utils import credit
from deeponto.data.infer.infer_utils import load_inference_dataset

CUR_TEMPLATE = None
CUR_VERBALIZER = None


@credit("openprompt", "https://github.com/thunlp/OpenPrompt/blob/main/experiments/cli.py")
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
        teacher_forcing=config[split].teacher_forcing
        if hasattr(config[split], "teacher_forcing")
        else None,
        predict_eos_token=True if config.task == "generation" else False,
        **config.dataloader,
    )
    example = template.incorporate_text_example(random.choice(dataset))
    logger = logging.getLogger()
    logger.info(f"transformed example: {example}")
    return dataloader


@credit("openprompt", "https://github.com/thunlp/OpenPrompt/blob/main/experiments/cli.py")
def run_inference(config, args):
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
    train_dataset, valid_dataset, test_dataset, Processor = load_inference_dataset(
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
            raise ValueError(
                "use few_shot setting but config.few_shot.few_shot_sampling is not specified"
            )
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
        template = load_template(
            config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config
        )
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
        template = load_template(
            config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config
        )
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
