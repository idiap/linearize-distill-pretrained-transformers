#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>; Authors of open-instruct (see https://github.com/allenai/open-instruct)
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#


#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import random
import datasets
from datetime import timedelta
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import deepspeed
import sys
import signal
from pile_dataset import MMapIndexedDataset
from train_utils import save_with_accelerate, ValueWindow
import json
from collections import defaultdict
from torch.nn import functional as F

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftConfig

logger = get_logger(__name__)


# try:
#     from hf_olmo import OLMoTokenizerFast
# except ImportError:
#     logger.warning("OLMo not installed. Ignore if using a different model.")

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument("--max_train_samples", type=int, default=7320644)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--model_revision",
        help="""If given, specifies a model revision (for HuggingFace models). This will 
        be applied to both the `model_name_or_path` and `config_name` args.""",
        default="main",
        required=False,
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--mixed_precision", default=None
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_revision",
        help="""Specifies a revision for the tokenizer. If not given, defaults
             to the value of the `model_revision` arg. In most cases, the tokenizer
             revision should be the same as the model revision and this flag shouldn't
             be needed.""",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2049,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--no_decay", type=str, default="bias,norm.weight")
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
        default=1,
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
        "--warmup_ratio", type=float, default=0.03, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--resume_run_id",
        type=str,
        default=None,
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
        "--run_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Turn on gradient checkpointing. Saves memory but slows training."
        ),
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help=(
            "Use qLoRA training - main thing is initialising model in quantised form. Not compatible with deepspeed."
        ),
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=-1,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--use_8bit_optimizer',
        action='store_true',
        help='Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--add_bos',
        action='store_true',
        help='Forcibly add bos token to the beginning of the input sequence. Use only when tokenizer does not add bos token by default (e.g., olmo).',
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=1800,
        help='Timeout for the training process. Useful if tokenization process is long. Default is 1800 seconds (30 minutes).',
    )
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        help='Trust remote code when loading pretrained models and tokenizers. Use only when you trust the remote code.',
    )
    parser.add_argument(
        '--reduce_loss',
        default='mean',
        choices=['mean', 'sum'],
        help='How to reduce loss over tokens. Default is mean, but using sum can improve chat model performance.',
    )
    parser.add_argument(
        '--wandb_entity',
        type=str,
        default=None,
        help='Entity to use for logging to wandb.'
    )
    parser.add_argument(
        '--use_mamba',
        action='store_true',
    )
    parser.add_argument(
        '--use_mamba2',
        action='store_true',
    )
    parser.add_argument(
        '--mamba_norm',
        type=str, default='none',
    )
    parser.add_argument(
        '--mamba_drop_nonfinite',
        action='store_true',
    )
    parser.add_argument('--weight_decay_mamba_only', action='store_true')
    parser.add_argument('--use_tf32', action='store_true')

    parser.add_argument('--beta2', type=float, default=0.999)

    parser.add_argument('--teacher_name_or_path', type=str, default=None)
    parser.add_argument(
        "--student_reinit", action="store_true"
    )
    parser.add_argument('--alpha_layer_distill_loss', type=float, default=10)
    parser.add_argument('--alpha_logits_distill_loss', type=float, default=0)
    parser.add_argument('--alpha_ce_loss', type=float, default=1)
    parser.add_argument('--freeze_steps', type=int, default=0)
    parser.add_argument('--distill_normalization', type=str, default="layernorm")
    parser.add_argument('--distill_position', type=str, default="block")
    parser.add_argument('--distill_skip_last', type=int, default=1)
    parser.add_argument('--distill_loss_type', type=str, default="mse")

    parser.add_argument('--lr_decay_steps', type=int, default=None)
    parser.add_argument('--min_lr', type=float, default=0)

    args = parser.parse_args()
    if (args.output_dir is not None and args.run_name is not None
            and os.path.isdir(args.output_dir) and args.output_dir[-1] == '/'):
        args.output_dir = os.path.join(args.output_dir, args.run_name)

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    return args

def main():
    args = parse_args()
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs]
    )
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

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    tokenizer_revision = (
        args.model_revision
        if args.tokenizer_revision is None
        else args.tokenizer_revision
    )

    if tokenizer_revision != args.model_revision:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tokenizer_revision}` is different
                   from the model revision `{args.model_revision}`."""
        logger.warn(warning)
    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            base_dir = args.output_dir if args.output_dir is not None else os.getcwd()
            # Get the most recent checkpoint
            dirs = [f.path for f in os.scandir(base_dir) if f.is_dir() and ("step_" in f.name or "epoch_" in f.name)]
            dirs.sort(key=lambda x: int(x.split("_")[-1]))
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        logger.info(f"Will resume from checkpoint: {checkpoint_path}")
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]
        do_resume = True
    else:
        do_resume = False

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            device_map = {"": device_index}  # force data-parallel training.
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                load_in_4bit=True,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True if args.use_flash_attn else False,
                revision=args.model_revision
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=args.trust_remote_code,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                use_flash_attention_2=True if args.use_flash_attn else False,
                revision=args.model_revision
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)
    logger.info(f"Total number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.teacher_name_or_path:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.teacher_name_or_path,
            use_flash_attention_2=True if args.use_flash_attn else False,
        )
        teacher_model.to(accelerator.device)
        teacher_model.train()

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0,
                                    1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    # elif (isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, GPT2TokenizerFast)) and isinstance(model, OPTForCausalLM):
    #     num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})
    # elif isinstance(tokenizer, OLMoTokenizerFast):
    #     only the eos for olmo, but we use it as bos
        # tokenizer.bos_token = tokenizer.eos_token
        # assert args.add_bos, "For OLMo, you must add bos token to the beginning of the input sequence."
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
        if len(tokenizer) > embeddings.weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))

    original_state = model.state_dict()

    if args.use_mamba2:
        args.use_mamba = True
    if args.use_mamba:
        if args.use_mamba2:
            import open_instruct.modeling_hybrid_mamba2 as mamba_utils
        else:
            import open_instruct.modeling_hybrid_mamba as mamba_utils
        model.config.mamba_norm = args.mamba_norm
        model.config.mamba_drop_nonfinite = args.mamba_drop_nonfinite
        model = mamba_utils.build_gpt_neox_mamba_model_from_gpt_neox(model)
        logger.info(f"Total number of parameters after substitution: {sum(p.numel() for p in model.parameters()):,}")

    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj",
                            "dense_h_to_4h", "dense_4h_to_h", "dense", "query_key_value"]
        )
        if args.output_dir is not None: # We have to save it separately, because to_bettertransformer() will unwrap the PeftModel
            peft_config.save_pretrained(os.path.join(args.output_dir, 'peft_config'))
        model = get_peft_model(model, peft_config)
        for name, param in model.named_parameters():
            if len(param.shape) == 1:
                param.requires_grad = True
        model.print_trainable_parameters()

    if not args.student_reinit and args.teacher_name_or_path:
        logger.info("Initializing student model from teacher weights")
        model.load_state_dict(teacher_model.state_dict(), strict=False)
    else:
        model.load_state_dict(original_state, strict=False)
        logger.info("Initializing student model from original weights")

    if do_resume:
        resume_state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
        if args.use_lora:
            resume_state_dict = {'base_model.model.' + k: v for k, v in resume_state_dict.items()}
        model.load_state_dict(resume_state_dict, strict=False)
        del resume_state_dict

    del original_state

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if args.use_lora:
            model.enable_input_require_grads()

    if not args.use_mamba:
        model = model.to_bettertransformer()
    if args.teacher_name_or_path:
        teacher_model = teacher_model.to_bettertransformer()


    train_dataset = MMapIndexedDataset(args.train_file, max_samples=args.max_train_samples,
                                       max_seq_length=args.max_seq_length)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        pin_memory=True,
        num_workers=4
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = args.no_decay.split(',')
    if args.weight_decay_mamba_only:
        no_decay += [n for n, p in model.named_parameters() if 'mamba' not in n]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    if args.use_qlora:
        from bitsandbytes.optim import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                                      betas=(0.9, args.beta2))

    if args.freeze_steps > 0:
        flipped_grads = []
        for name, param in model.named_parameters():
            if param.requires_grad and 'mamba' not in name:
                param.requires_grad = False
                flipped_grads.append(param)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    if args.lr_decay_steps is not None:
        num_training_steps_for_scheduler = args.lr_decay_steps
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )
    if args.min_lr > 0:
        for i, fn in enumerate(lr_scheduler.lr_lambdas):
            fn_ = lambda x: max(fn(x), args.min_lr / args.learning_rate) if x < num_training_steps_for_scheduler else args.min_lr / args.learning_rate
            lr_scheduler.lr_lambdas[i] = fn_
    if do_resume:
        if os.path.exists(os.path.join(checkpoint_path, "scheduler.pt")):
            scheduler_state_dict = torch.load(os.path.join(checkpoint_path, "scheduler.pt"))
            lr_scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict
            scheduler_loaded = True
        else:
            scheduler_loaded = False
        if os.path.exists(os.path.join(checkpoint_path, "optimizer.pt")):
            optimizer_state_dict = torch.load(os.path.join(checkpoint_path, "optimizer.pt"))
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None:
        if checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)
        else:
            try:
                checkpointing_steps = float(checkpointing_steps)
                checkpointing_steps = int(checkpointing_steps * args.max_train_steps)
            except:
                pass

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
            kwargs["wandb"]["tags"] = ["distill"]
        if args.resume_run_id:
            kwargs['wandb']["id"] = args.resume_run_id
            kwargs['wandb']["resume"] = "must"
        accelerator.init_trackers("retrain_pile_mamba",
                                  experiment_config,
                                  init_kwargs=kwargs)

    logger.info(str(model))

    for name, param in model.named_parameters():
        logger.info(f"{name}: {param.shape} {param.requires_grad}")

    # Train!

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Step per epoch = {num_update_steps_per_epoch}")
    logger.info(f"  Batch size = {args.per_device_train_batch_size}")
    logger.info(f"  Total estimated train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Checkpointing steps = {checkpointing_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save

    if do_resume:
        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                    int(training_difference.replace("step_", ""))
                    * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)
        if not scheduler_loaded:
            # We need to step the scheduler to the last step.
            for _ in range(completed_steps):
                lr_scheduler.step()

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    if args.distill_loss_type == 'mse':
        distill_loss = torch.nn.functional.mse_loss
    elif args.distill_loss_type == 'cosine':
        @torch.compile
        def cosine_loss(x, y):
            return 1 - torch.nn.functional.cosine_similarity(x, y, -1).mean()
        distill_loss = cosine_loss

    def signal_handler(sig, frame):
        if args.with_tracking:
            accelerator.end_training()

            output_dir = f"step_{completed_steps}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            save_with_accelerate(accelerator, model, tokenizer, optimizer, lr_scheduler, output_dir, args)
        sys.exit(0)

    signal.signal(signal.SIGTSTP, signal_handler)

    windows = defaultdict(lambda : ValueWindow(10))

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if (
                args.resume_from_checkpoint
                and epoch == starting_epoch
                and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    if args.teacher_name_or_path:
                        with torch.no_grad():
                            teacher_outputs_with_hidden = teacher_model(
                                input_ids=batch.data['input_ids'][:, :-1],
                                output_hidden_states=args.distill_position == 'block',
                                output_attentions=args.distill_position != 'block')

                        teacher_outputs_with_hidden = (
                            None,
                            teacher_outputs_with_hidden.logits.detach(),
                            [x.detach() for x in (teacher_outputs_with_hidden.hidden_states
                                                  if args.distill_position == 'block'
                                                  else teacher_outputs_with_hidden.attentions)],
                        )


                    try:
                        outputs = model(input_ids=batch.data['input_ids'][:, :-1],
                                        labels=batch.data['input_ids'][:, 1:],
                                        use_cache=False,
                                        no_shift_for_loss=True,
                                        output_hidden_states=args.distill_position == 'block',
                                        output_attentions=args.distill_position != 'block')
                        train_loss = outputs.loss
                        loss = train_loss * args.alpha_ce_loss

                        if args.alpha_layer_distill_loss != 0:
                            from train_utils import layer_distill_loss
                            base_loss, layer_loss = layer_distill_loss(
                                teacher_outputs_with_hidden[2], outputs.hidden_states,
                                distill_loss, 1, None,
                                args.distill_skip_last, args.distill_normalization)

                            loss = loss + args.alpha_layer_distill_loss * layer_loss
                            windows['base_distill_loss'].append(base_loss.item())
                            windows['distill_loss'].append(layer_loss.item())

                        if args.alpha_logits_distill_loss != 0:
                            temparature = 2.0
                            logits_distill_loss = F.kl_div(
                                input=F.log_softmax(outputs.logits / temparature, dim=-1),
                                target=F.softmax(teacher_outputs_with_hidden[1] / temparature, dim=-1),
                                reduction="none"
                            ).sum(-1).mean()
                            windows['logit_distill_loss'].append(logits_distill_loss.item())
                            loss = loss + args.alpha_logits_distill_loss * logits_distill_loss

                    except Exception as e:
                        logger.warn(f"Error in step {step} of epoch {epoch}")
                        logger.warn(f"Batch shape: {batch['input_ids'].shape}")
                        raise e

                    windows['loss'].append(train_loss.item())
                    windows['total_loss'].append(loss.item())

                    if args.teacher_name_or_path:
                        del teacher_outputs_with_hidden
                    del outputs
                    if not torch.isfinite(loss):
                        logger.warn("Loss is not finite. Skipping this batch.")
                        del loss
                        continue

                    accelerator.backward(loss)
                    # clip gradient norm. don't do this with deepspeed
                    if accelerator.sync_gradients and args.clip_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    del loss

                    # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                # ctx.step()
                for name in windows:
                    windows[name].commit()
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    desc = f"L={windows['total_loss'].average:.3f} "
                    desc += f"(c {windows['loss'].average:.3f} "
                    if args.alpha_layer_distill_loss != 0:
                        desc += f"d {windows['distill_loss'].average:.3f} "
                    if args.alpha_logits_distill_loss != 0:
                        desc += f"l {windows['logit_distill_loss'].average:.3f} "
                    desc += f") lr={lr_scheduler.get_last_lr()[0]:.2e}"
                    progress_bar.desc = desc

                    if args.with_tracking:
                        log_dict = {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": windows['loss'].average,
                                "total_loss": windows['total_loss'].average,
                                "epoch": epoch,
                            }
                        if args.alpha_layer_distill_loss != 0:
                            log_dict.update({
                                "distill_loss": windows['distill_loss'].average,
                                "base_distill_loss": windows['base_distill_loss'].average,
                            })
                        if args.alpha_logits_distill_loss != 0:
                            log_dict.update({
                                "logit_distill_loss": windows['logit_distill_loss'].average,
                            })
                        accelerator.log(
                            log_dict,
                            step=completed_steps
                        )

                if completed_steps == args.freeze_steps:
                    for param in flipped_grads:
                        param.requires_grad = True
                    logger.info("Unfreezing the model")

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        save_with_accelerate(accelerator, model, tokenizer, optimizer, lr_scheduler, output_dir, args)

                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            save_with_accelerate(accelerator, model, tokenizer, optimizer, lr_scheduler, output_dir, args)

    if args.output_dir is not None:
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        save_with_accelerate(accelerator, model, tokenizer, optimizer, lr_scheduler, args.output_dir, args)

    accelerator.wait_for_everyone()
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()

