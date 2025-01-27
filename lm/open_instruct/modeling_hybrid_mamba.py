#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>; state-spaces/mamba org and HuggingFace Inc. team.
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#


# Copyright 2024 state-spaces/mamba org and HuggingFace Inc. team.
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
"""PyTorch MAMBA model."""

from open_instruct.hf_modelling_mamba import *
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXMLP


class HybridMambaBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mamba = MambaMixer(config, layer_idx=layer_idx)

    def forward(self, hidden_states, cache_params: Optional[MambaCache] = None):
        hidden_states = self.mamba(hidden_states, cache_params=cache_params)
        return hidden_states

class GPTNeoXMambaLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
        self.attention = HybridMambaBlock(config, layer_idx)
        self.mlp = GPTNeoXMLP(config.gpt_neox_config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor], cache_params: Optional[MambaCache] = None,
    ):
        attention_layer_outputs = self.attention(
            self.input_layernorm(hidden_states),
            cache_params=cache_params
        )
        attn_output = attention_layer_outputs
        attn_output = self.post_attention_dropout(attn_output)

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output

        return hidden_states

class GPTNeoXMambaModel(MambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([GPTNeoXMambaLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_in

    def set_input_embeddings(self, new_embeddings):
        self.embed_in = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[MambaCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # `attention_mask` is passed by the tokenizer and we don't want it
    ) -> Union[Tuple, MambaOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)
        hidden_states = self.emb_dropout(inputs_embeds)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if cache_params is None and use_cache:
            cache_params = MambaCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(mixer_block.__call__, hidden_states, cache_params)
            else:
                hidden_states = mixer_block(hidden_states, cache_params=cache_params)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )

class HybridMambaForCausalLM(MambaPreTrainedModel):
    base_model_prefix = "gpt_neox"
    _no_split_modules = ["HybridMambaBlock"]
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.gpt_neox = GPTNeoXMambaModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    def get_input_embeddings(self):
        return self.gpt_neox.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.gpt_neox.set_input_embeddings(new_embeddings)

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params: Optional[MambaCache] = None,
        **kwargs,
    ):
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "cache_params": cache_params,
                "use_cache": use_cache,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        no_shift_for_loss=False,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, MambaCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba_outputs = self.gpt_neox(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )
        hidden_states = mamba_outputs[0]

        logits = self.embed_out(hidden_states.to(self.embed_out.weight.dtype)).float()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if no_shift_for_loss:
                shift_logits = logits.contiguous()
                shift_labels = labels.contiguous()
            else:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MambaCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
        )

def build_gpt_neox_mamba_model(gpt_neox_config):
    mamba_config = MambaConfig.from_pretrained("state-spaces/mamba-2.8b-hf")
    mamba_config.hidden_size = gpt_neox_config.hidden_size
    mamba_config.use_parallel_residual = gpt_neox_config.use_parallel_residual
    mamba_config.layer_norm_eps = gpt_neox_config.layer_norm_eps
    mamba_config.hidden_dropout = gpt_neox_config.hidden_dropout
    mamba_config.vocab_size = gpt_neox_config.vocab_size
    mamba_config.use_mamba = True
    mamba_config.num_hidden_layers = mamba_config.n_layers = gpt_neox_config.num_hidden_layers
    mamba_config.max_position_embeddings = gpt_neox_config.max_position_embeddings
    mamba_config.tie_word_embeddings = False

    mamba_config.rms_norm = False
    mamba_config.gpt_neox_config = gpt_neox_config

    mamba_model = HybridMambaForCausalLM(mamba_config)
    return mamba_model

def build_gpt_neox_mamba_model_from_gpt_neox(gpt_neox_model):
    mamba_model = build_gpt_neox_mamba_model(gpt_neox_model.config)
    state_dict = gpt_neox_model.state_dict()
    mamba_model.load_state_dict(state_dict, strict=False)
    return mamba_model