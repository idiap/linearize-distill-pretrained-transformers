#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#


class ValueWindow():
    def __init__(self, window_size=100, accumulated=True):
        self._window_size = window_size
        self._values = []
        self._buffer = []
        self._accumulated = accumulated

    def append(self, x):
        if self._accumulated:
            self._buffer.append(x)
        else:
            self._values = self._values[-(self._window_size - 1):] + [x]

    def commit(self):
        if self._accumulated and self._buffer:
            self._values = self._values[-(self._window_size - 1):] + [sum(self._buffer) / len(self._buffer)]
            self._buffer = []

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []

import torch
import os
from torch import layer_norm
def save_with_accelerate(accelerator, model, tokenizer, optimizer, scheduler, output_dir, args):
    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    params = {k: v for k, v in model.named_parameters()}
    state_dict = {k: v for k, v in state_dict.items() if params[k].requires_grad}
    os.makedirs(output_dir, exist_ok=True)
    if args.use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            state_dict = {(k[len("base_model.model."):] if k.startswith("base_model.model.") else k): v for k, v in state_dict.items()}
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
    else:
        # don't use safetensors for saving for now
        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False, max_shard_size="20GB"
        )
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
    accelerator.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    accelerator.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    model.config.save_pretrained(output_dir)


def layer_distill_loss(all_teacher_features, all_student_features, distill_loss, distill_scale_factor, mask,
                       distill_skip_last, distill_normalization):
    layer_distill_losses = []
    base_distill_losses = []
    for i in range(1, len(all_teacher_features) - distill_skip_last):
        teacher_features = all_teacher_features[i]
        student_features = all_student_features[i]
        if mask is not None:
            teacher_features = teacher_features * mask
            student_features = student_features * mask
        t_layer_distill_loss = distill_loss(teacher_features, student_features) * distill_scale_factor
        base_distill_losses.append(t_layer_distill_loss.detach())
        if distill_normalization == 'layernorm':
            teacher_features = layer_norm(teacher_features, [teacher_features.shape[-1]])
            student_features = layer_norm(student_features, [student_features.shape[-1]])
            t_layer_distill_loss = distill_loss(teacher_features, student_features) * distill_scale_factor
        elif distill_normalization == 'rmsnorm':
            teacher_features = teacher_features / torch.sqrt((teacher_features ** 2).mean(-1, keepdim=True) + 1e-6)
            student_features = student_features / torch.sqrt((student_features ** 2).mean(-1, keepdim=True) + 1e-6)
            t_layer_distill_loss = distill_loss(teacher_features, student_features) * distill_scale_factor
        elif distill_normalization == 'loss_norm':
            t_layer_distill_loss = distill_loss(teacher_features, student_features)
            t_layer_distill_loss = t_layer_distill_loss / t_layer_distill_loss.detach()
            t_layer_distill_loss *= distill_scale_factor
        layer_distill_losses.append(t_layer_distill_loss)
    layer_loss = torch.stack(layer_distill_losses).mean()
    return torch.stack(base_distill_losses).mean(), layer_loss
