#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#

from transformers import Wav2Vec2ForCTC, BartForConditionalGeneration, T5ForConditionalGeneration
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, BartTokenizer, T5Tokenizer
from transformers.utils import ModelOutput
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
import logging
import math
from models.mamba_utils import substitute_layer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2048, pe_weight=0.1, mode='encoding'):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_weight = pe_weight
        if mode == 'encoding':
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        elif mode == 'embedding':
            self.pe = nn.Parameter(torch.empty(max_len, d_model))
            nn.init.normal_(self.pe)

    def forward(self, x):
        x = x + self.pe_weight * self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)

class Adaptor(nn.Module):
    def __init__(self, hp, in_size, out_size):
        super().__init__()
        self.hparams = hp
        self.in_size = in_size
        self.out_size = out_size
        if hp.adaptor_pos_encoding:
            self.pe = PositionalEncoding(out_size, mode=hp.adaptor_pos_encoding_mode, pe_weight=hp.adaptor_pe_weight)
        if hp.adaptor_type == 'conv':
            self.conv = nn.ModuleList()
            self.layernorm = nn.ModuleList()
            stride = 2
            conv_out_size = out_size * 2 if hp.adaptor_use_glu else out_size
            for i in range(hp.adaptor_num_layers):
                self.conv.append(nn.Conv1d(in_size, conv_out_size, kernel_size=3, stride=stride, padding=1))
                if hp.adaptor_use_layernorm:
                    self.layernorm.append(nn.LayerNorm(out_size))
                in_size = out_size
                stride = 2
        elif hp.adaptor_type == 'linear':
            self.linear = nn.Linear(in_size, out_size, bias=False)
        elif hp.adaptor_type == 'none':
            assert in_size == out_size
        else:
            raise NotImplementedError('adaptor_type {} is not implemented'.format(self.hparams.adaptor_type))

    def forward(self, inputs):
        if self.hparams.adaptor_type == 'conv':
            inputs = inputs.transpose(1, 2)
            for i in range(self.hparams.adaptor_num_layers):
                inputs = self.conv[i](inputs)
                if self.hparams.adaptor_use_glu:
                    inputs = F.glu(inputs, dim=1)
                else:
                    inputs = F.relu(inputs)
                if self.hparams.adaptor_use_layernorm:
                    # inputs = inputs.transpose(1, 2)
                    inputs = self.layernorm[i](inputs)
            inputs = inputs.transpose(1, 2)
            if self.hparams.adaptor_pos_encoding:
                inputs = self.pe(inputs)
            return inputs
        elif self.hparams.adaptor_type == 'linear':
            inputs = self.linear(inputs)
            return inputs
        elif self.hparams.adaptor_type == 'none':
            return inputs

    def transform_attention_mask(self, mask):
        if self.hparams.adaptor_type == 'conv':
            right = mask.sum(-1) - 1
            n = mask.shape[-1] - 1
            for i in range(self.hparams.adaptor_num_layers):
                right = right // 2
                n = n // 2

            mask_ = torch.zeros(mask.shape[0], n + 1, dtype=torch.bool, device=mask.device)
            mask_[(torch.arange(mask.shape[0], device=mask_.device), right)] = True
            mask_ = mask_.flip([-1]).cumsum(-1).flip([-1]).bool()

            return mask_
        elif self.hparams.adaptor_type in ['linear', 'none']:
            return mask

class Identity(nn.Identity):
    def forward(self, input, **kwargs):
        return ModelOutput(last_hidden_state=input, hidden_states=[input], attentions=None)

class Classifier(nn.Module):
    def __init__(self, hidden_size, num_classes, pooling_type):
        super().__init__()
        if not isinstance(num_classes, list):
            num_classes = [num_classes]
        self.pooling_type = pooling_type
        self.linear = nn.ModuleList([nn.Linear(hidden_size, n) for n in num_classes])

    def forward(self, hidden_states, input_masks):
        if self.pooling_type == 'mean':
            features = (hidden_states * input_masks.unsqueeze(-1)).sum(1) / input_masks.sum(-1).unsqueeze(-1)
        elif self.pooling_type == 'first':
            features = hidden_states[:, 0]
        logits = [linear(features) for linear in self.linear]
        return logits


def build_asr_model(hparams):
    if hparams.asr_model_type == 'ctc':
        asr_model = Wav2Vec2ForCTC.from_pretrained(hparams.asr_model_name,
                                                   ctc_loss_reduction="none",
                                                   pad_token_id=hparams.pad_token_id,
                                                   ctc_zero_infinity=True,
                                                   # Even we manually mask out inf, the gradients will still be nan
                                                   feat_proj_dropout=hparams.feat_proj_dropout,
                                                   final_dropout=hparams.final_dropout,
                                                   hidden_dropout=hparams.hidden_dropout,
                                                   activation_dropout=hparams.activation_dropout,
                                                   mask_time_prob=hparams.mask_time_prob,
                                                   vocab_size=hparams.asr_vocab_size,
                                                   attn_implementation="flash_attention_2" if hparams.asr_use_flash_attn else None,
                                                   )
    else:
        raise NotImplementedError('asr_model_type {} is not implemented'.format(hparams.asr_model_type))
    # asr_model.to_bettertransformer()
    asr_model.freeze_feature_encoder()

    if hparams.asr_keep_feature_encoder_only:
        asr_model.wav2vec2.encoder = Identity()

    logging.info("ASR model built, with total %.2fM parameters", asr_model.num_parameters() / 1e6)

    if hparams.asr_use_mamba:
        substitute_layer(asr_model.wav2vec2.encoder, asr_model.config,
                         use_mamba2=hparams.asr_mamba_use_mamba2, residual_inside=hparams.asr_mamba_residual_inside)
        logging.info("Mamba layer substituted, now total %.2fM parameters", asr_model.num_parameters() / 1e6)

    return asr_model

def build_text_decoder(hparams):
    if not hparams.decoder_model_name:
        hparams.decoder_model_name = hparams.decoder_name
    if hparams.decoder_type == 'bart':
        decoder = BartForConditionalGeneration.from_pretrained(hparams.decoder_model_name,
                                                               dropout=hparams.decoder_dropout,
                                                               activation_dropout=hparams.decoder_dropout)
        decoder.config.max_length = 200
        if hasattr(decoder.model, 'encoder') and not hparams.decoder_stack_encoder:
            del decoder.model.encoder
    elif hparams.decoder_type == 't5':
        decoder = T5ForConditionalGeneration.from_pretrained(hparams.decoder_model_name)
        decoder.config.max_length = 200
        del decoder.encoder
    else:
        raise NotImplementedError('decoder_type {} is not implemented'.format(hparams.decoder_type))
    logging.info("Text decoder built, with total %.2fM parameters", decoder.num_parameters() / 1e6)
    return decoder

def build_processor(hp, vocab_path):
    processor = defaultdict(None)
    if hp.asr_processor_name:
        processor['tokenizer'] = Wav2Vec2CTCTokenizer.from_pretrained(hp.asr_processor_name)
        processor['extractor'] = Wav2Vec2FeatureExtractor.from_pretrained(hp.asr_processor_name)
    else:
        processor['tokenizer'] = Wav2Vec2CTCTokenizer(vocab_path, unk_token='[UNK]', pad_token="[PAD]", word_delimiter_token="|")
        processor['extractor'] = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                             return_attention_mask=hp.use_attention_mask)
    if hp.use_decoder:
        if hp.decoder_type == 'bart':
            processor['decoder_tokenizer'] = BartTokenizer.from_pretrained(hp.decoder_name)
        elif hp.decoder_type == 't5':
            processor['decoder_tokenizer'] = T5Tokenizer.from_pretrained(hp.decoder_name)
    else:
        processor['decoder_tokenizer'] = None

    return processor