#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>; Authors of LoSparse (See https://github.com/yxli2123/LoSparse); The HuggingFace Inc. team
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#


# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os, shutil
import random
from pathlib import Path
import sys, signal

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from torch.nn import MSELoss
import torch.nn.functional as F
from train_utils import ValueWindow
from linformer_utils import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.


logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "imdb": ("text", None),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_group_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=-1, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument('--keep_last_best_ckpt', action='store_true')
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--teacher_path",
        type=str
    )
    parser.add_argument(
        "--teacher_ckpt",
        type=str, default=None
    )
    parser.add_argument(
        "--student_reinit", action="store_true"
    )
    parser.add_argument("--alpha_output", type=int, default=1)
    parser.add_argument("--alpha_layer", type=int, default=1)
    parser.add_argument("--freeze_steps", type=int, default=-1)
    parser.add_argument("--eval_interval", type=int, default=-1)
    parser.add_argument("--eval_checkpoint", type=str, default="No checkpoint", help="use this during the evaluation")
    parser.add_argument("--no_substitute", action="store_true")
    parser.add_argument("--proj_share_mode", type=str, default="none")
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument('--max_distill_steps', type=int, default=None)
    args = parser.parse_args()

    if args.config_file is not None:
        config = json.load(open(args.config_file))
        for key, value in config.items():
            if not hasattr(args, key):
                raise ValueError(f"Invalid arg: {key}")
            setattr(args, key, value)

    if args.project_name is None:
        args.project_name = 'linformer_distill' + '_' + args.task_name

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()
    accelerator = Accelerator(log_with=args.report_to)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_group_name is not None:
        raw_datasets = datasets.load_dataset(args.dataset_group_name + '/' + args.task_name)
    elif args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        raw_datasets = None
    # Loading the dataset from local csv or json file.
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if data_files:
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        n_raw_datasets = load_dataset(extension, data_files=data_files)
    else:
        n_raw_datasets = None
    if raw_datasets is None:
        raw_datasets = n_raw_datasets
    elif n_raw_datasets:
        for split in n_raw_datasets.keys():
            raw_datasets[split] = n_raw_datasets[split]

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    ####Prepare teacher model
    if args.teacher_path is not None:
        temperature = 2
        teacher_path = args.teacher_path

        t_config = AutoConfig.from_pretrained(teacher_path, num_labels=num_labels, finetuning_task=args.task_name)
        teacher_model = AutoModelForSequenceClassification.from_pretrained(
            teacher_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=t_config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
        if args.teacher_ckpt is not None:
            teacher_model.load_state_dict(torch.load(args.teacher_ckpt))
        teacher_model.to(accelerator.device)
        teacher_model.train()

    ###Prepare student model
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )
    original_state = model.state_dict()

    if not args.no_substitute:
        substitute_layer(model, config, args.proj_share_mode)
        if not args.student_reinit:
            model.load_state_dict(teacher_model.state_dict(), strict=False)
        else:
            model.load_state_dict(original_state, strict=False)
            logger.info("Initializing student model from original weights")
    if args.freeze_steps > 0:
        for name, param in model.named_parameters():
            if not ('proj_k' in name or 'proj_v' in name):
                param.requires_grad = False
    model = model.to(accelerator.device)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(accelerator.device)
    if args.eval_checkpoint != "No checkpoint":
        print(model.load_state_dict(
            torch.load(args.eval_checkpoint, map_location=accelerator.device),
            strict=False))
    else:
        print("Not doing eval")
    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and args.task_name is not None
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    val_name = {"mnli": "validation_matched", "mnli-mm": "validation_mismatched", "imdb": "test"}.get(args.task_name, "validation")
    eval_dataset = processed_datasets[val_name]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    if args.eval_interval == -1:
        args.eval_interval = math.ceil(num_update_steps_per_epoch / 4)
    if args.num_warmup_steps == -1:
        args.num_warmup_steps = int(0.04 * args.max_train_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        kwargs = {"wandb": {}}
        if args.run_name:
            kwargs["wandb"]["name"] = args.run_name
            kwargs["wandb"]["group"] = args.run_name[: args.run_name.rfind(".")]
        accelerator.init_trackers(args.project_name, experiment_config, init_kwargs=kwargs)

    # Get the metric function
    if args.task_name is not None and args.dataset_group_name is None:
        metric = evaluate.load("glue", args.task_name)
    else:
        metric = evaluate.load("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Num eval examples = {len(eval_dataset)}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        epoch_steps = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            completed_steps = starting_epoch * epoch_steps
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            completed_steps = resume_step
            starting_epoch = resume_step // epoch_steps
            resume_step -= starting_epoch * epoch_steps

    alpha_output_distil_loss = args.alpha_output
    alpha_layer_distill_loss = args.alpha_layer
    loss_mse = MSELoss()

    loss_window = ValueWindow(10)
    ce_loss_window = ValueWindow(10)
    distill_loss_window = ValueWindow(10)

    best_metric = {'best_step': 0, 'best_metric': 0, 'results': []}

    # Save results if we receive a Ctrl+Z
    def signal_handler(sig, frame):
        if args.output_dir is not None:
            with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                json.dump(best_metric, f)
        if args.with_tracking:
            accelerator.end_training()
        sys.exit(0)

    progress_bar.update(completed_steps)

    signal.signal(signal.SIGTSTP, signal_handler)
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            do_distill = (args.max_distill_steps is None or completed_steps < args.max_distill_steps) and args.teacher_path is not None
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue
            outputs_with_hidden = model(**batch, output_hidden_states=True)
            if do_distill:
                teacher_inputs = {}
                teacher_inputs_ = batch.copy()
                for k, v in teacher_inputs_.items():
                    teacher_inputs[k] = v.detach().clone()
                with torch.no_grad():
                    teacher_outputs_with_hidden = teacher_model(**teacher_inputs, output_hidden_states=True)

                teacher_outputs_with_hidden = (teacher_outputs_with_hidden[0].detach(),
                                               teacher_outputs_with_hidden[1].detach(),
                                               [x.detach() for x in teacher_outputs_with_hidden[2]])

            loss = outputs_with_hidden.loss
            ce_loss_window.append(loss.item())
            if do_distill and alpha_output_distil_loss != 0:
                distillation_loss = F.kl_div(
                    input=F.log_softmax(outputs_with_hidden[1] / temperature, dim=-1),
                    target=F.softmax(teacher_outputs_with_hidden[1].detach() / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature ** 2)
                loss += alpha_output_distil_loss * distillation_loss
            layer_distill_loss = 0
            if do_distill and alpha_layer_distill_loss != 0:
                distill_losses = []
                for i in range(13):
                    distill_losses.append(loss_mse(outputs_with_hidden[2][i], teacher_outputs_with_hidden[2][i].detach()))
                    layer_distill_loss += distill_losses[-1]
                distill_loss_window.append(layer_distill_loss.item())
                loss += alpha_layer_distill_loss * layer_distill_loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss_window.append(loss.item())
            progress_bar.desc = (f"loss: {loss_window.average:.4f} "
                                 f"({ce_loss_window.average:.4f} {distill_loss_window.average:.4f}) "
                                 f"lr: {optimizer.param_groups[0]['lr']:.4e}")
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps == args.freeze_steps:
                    for param in model.parameters():
                        param.requires_grad = True
                    logger.info("Unfreezing the model")

                if completed_steps % 10 == 0 and args.with_tracking:
                    accelerator.log(
                        {
                            "step": completed_steps,
                            "epoch": epoch,
                            "loss": loss_window.average,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "ce_loss": ce_loss_window.average,
                            "distill_loss": distill_loss_window.average,
                        },
                        step=completed_steps
                    )

                if completed_steps % args.eval_interval == 0 or step == len(train_dataloader) - 1:
                    model.eval()
                    samples_seen = 0
                    for step, batch in enumerate(eval_dataloader):
                        with torch.no_grad():
                            outputs = model(**batch)
                        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                        predictions, references = accelerator.gather((predictions, batch["labels"]))
                        # If we are in a multiprocess environment, the last batch has duplicates
                        if accelerator.num_processes > 1:
                            if step == len(eval_dataloader) - 1:
                                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                                references = references[: len(eval_dataloader.dataset) - samples_seen]
                            else:
                                samples_seen += references.shape[0]
                        metric.add_batch(
                            predictions=predictions,
                            references=references,
                        )

                    eval_metric = metric.compute()
                    model.train()
                    logger.info(f"Step {completed_steps}:  {eval_metric}")

                    if args.with_tracking:
                        accelerator.log(
                            {
                                "accuracy" if args.task_name is not None and args.dataset_group_name is None else "glue": eval_metric,
                            },
                            step=completed_steps,
                        )

                    best_metric['results'].append((completed_steps, eval_metric))
                    if list(eval_metric.values())[0] >= best_metric['best_metric']:
                        logger.info(f"Update best metric from {best_metric['best_metric']} to {list(eval_metric.values())[0]}")
                        best_metric['best_step'] = completed_steps
                        best_metric['best_metric'] = list(eval_metric.values())[0]
                        output_dir = f"best_step_{completed_steps}"
                        if args.keep_last_best_ckpt:
                            for f in os.listdir(args.output_dir):
                                if f.startswith("best_step"):
                                    shutil.rmtree(os.path.join(args.output_dir, f))
                        if args.output_dir is not None and args.save_checkpoint:
                            output_dir = os.path.join(args.output_dir, output_dir)
                            accelerator.save_state(output_dir, safe_serialization=False)

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None and args.save_checkpoint:
                        output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir, safe_serialization=False)

            if completed_steps >= args.max_train_steps:
                break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None and args.save_checkpoint:
                output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir, safe_serialization=False)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None and args.save_checkpoint:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save,
            safe_serialization=False
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")

    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(best_metric, f)


if __name__ == "__main__":
    main()
