#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#

import torch
from torch import nn
from mamba_ssm import Mamba2, Mamba

def get_reverse_padded_index(x_shape, lengths):
    indices = torch.arange(x_shape[1], device=lengths.device, dtype=lengths.dtype
                           ).repeat(x_shape[0], 1) + x_shape[1] - lengths.unsqueeze(1)
    indices = torch.clamp(indices, 0, x_shape[1])
    indices = indices.unsqueeze(-1).repeat([1, 1, x_shape[-1]])

    return indices

def reverse_padded_sequence(x, indices):
    x = torch.cat([x.flip(dims=[1]), torch.zeros_like(x[:, 0:1])], dim=1)
    return torch.gather(x, dim=1, index=indices)

class BiMamba2Wrapper(nn.Module):
    def __init__(
            self,
            d_model,
            norm='none',
            drop_non_finite=False,
            use_mamba2=True,
            residual_inside=False
    ):
        super().__init__()
        if use_mamba2:
            self.fw_mamba = Mamba2(d_model=d_model, expand=1)
            self.bw_mamba = Mamba2(d_model=d_model, expand=1)
        else:
            self.fw_mamba = Mamba(d_model=d_model, expand=1)
            self.bw_mamba = Mamba(d_model=d_model, expand=1)
        self.norm = norm
        if norm == 'layer':
            self.post_norm = nn.LayerNorm(d_model)
        self.drop_non_finite = drop_non_finite
        self.residual_inside = residual_inside

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        if attention_mask is not None:
            if len(attention_mask.size()) == 4:
                attention_mask = (attention_mask == 0).squeeze(1)[:, 0]
            padding_lengths = attention_mask.sum(dim=-1)
        else:
            padding_lengths = torch.ones(hidden_states.size(0),
                                     device=hidden_states.device, dtype=torch.long) * hidden_states.size(1)
        rev_indices = get_reverse_padded_index(hidden_states.size(), padding_lengths)
        reversed_hidden_states = reverse_padded_sequence(hidden_states, rev_indices)

        fw_outputs = self.fw_mamba(hidden_states)
        bw_outputs = self.bw_mamba(reversed_hidden_states)

        bw_outputs = reverse_padded_sequence(bw_outputs, rev_indices)

        if self.residual_inside:
            outputs = hidden_states + fw_outputs + bw_outputs
        else:
            outputs = fw_outputs + bw_outputs
        if attention_mask is not None:
            outputs = torch.where(attention_mask[..., None], outputs, torch.zeros_like(outputs))

        if self.drop_non_finite:
            outputs = torch.where(torch.isfinite(outputs), outputs, torch.zeros_like(outputs))
        if self.norm != 'none':
            outputs = self.post_norm(outputs)
        results = (outputs, outputs, outputs)
        return results

def substitute_layer(module, config, **kwargs):
    for name, subm in module.named_children():
        if name in ['attention']:
            new_module = BiMamba2Wrapper(config.hidden_size,
                                         use_mamba2=kwargs.get('use_mamba2', True),
                                         residual_inside=kwargs.get('residual_inside', False))
            setattr(module, name, new_module)
        else:
            substitute_layer(subm, config, **kwargs)