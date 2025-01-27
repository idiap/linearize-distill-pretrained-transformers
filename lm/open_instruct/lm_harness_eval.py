#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#

import torch

import transformers
from transformers import AutoTokenizer

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from eval.utils import load_hf_lm


@register_model("hybrid_mamba")
class HybridMambaEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained, **kwargs):
        model = load_hf_lm(pretrained)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
        super().__init__(pretrained=model, tokenizer=tokenizer, **kwargs)
