#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#

from torch import nn
import torch
import math

from typing import List, Optional, Tuple, Union

shared_proj_kv = None


class LinformerSelfAttention(nn.Module):
    def __init__(self, config, k=256, proj_share_mode='none'):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        if proj_share_mode == 'layer':
            global shared_proj_kv
            if shared_proj_kv is None:
                shared_proj_kv = nn.Parameter(torch.zeros(config.max_position_embeddings, k))
                nn.init.normal_(shared_proj_kv, std=1.0)
            self.proj_k = self.proj_v = shared_proj_kv
        else:
            self.proj_k = nn.Parameter(torch.zeros(config.max_position_embeddings, k))
            nn.init.normal_(self.proj_k, std=1.0)
            if proj_share_mode == 'none':
                self.proj_v = nn.Parameter(torch.zeros(config.max_position_embeddings, k))
                nn.init.normal_(self.proj_v, std=1.0)
            else:
                self.proj_v = self.proj_k

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        proj_k = self.proj_k
        proj_v = self.proj_v
        if hidden_states.shape[1] < self.proj_k.shape[0]:
            proj_k = self.proj_k[:hidden_states.shape[1]]
            proj_v = self.proj_v[:hidden_states.shape[1]]
        hk = self.key(hidden_states)
        hv = self.value(hidden_states)

        if attention_mask is not None:
            lengths = (attention_mask > -1).sum(-1).squeeze()
            if lengths.dim() == 0:
                lengths = lengths.unsqueeze(0)
            hk = torch.stack([torch.matmul(proj_k[:lengths[i]].transpose(-1, -2), hk[i, :lengths[i]])
                              / math.sqrt(lengths[i]) for i in range(lengths.shape[0])], dim=0)
            hv = torch.stack([torch.matmul(proj_v[:lengths[i]].transpose(-1, -2), hv[i, :lengths[i]])
                              / math.sqrt(lengths[i]) for i in range(lengths.shape[0])], dim=0)
        else:
            hk = torch.matmul(hk.transpose(-1, -2), proj_k).transpose(-1, -2) / math.sqrt(proj_k.shape[0])
            hv = torch.matmul(hv.transpose(-1, -2), proj_v).transpose(-1, -2) / math.sqrt(proj_v.shape[0])

        key_layer = self.transpose_for_scores(hk)
        value_layer = self.transpose_for_scores(hv)
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


def substitute_layer(module, config, proj_share_mode, **kwargs):
    for name, subm in module.named_children():
        if name in ['self']:
            new_module = LinformerSelfAttention(config, proj_share_mode=proj_share_mode)
            setattr(module, name, new_module)
        else:
            substitute_layer(subm, config, proj_share_mode, **kwargs)
