#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#

from utils.hparams import HParams

hparams = HParams(
    sr=16000,

    max_eval_batches=10000,

    data_format="nlt",
    bucket_size=1024,
    shuffle_training_data=True,
    batch_frame_limit=2.5e6,
    batch_quad_frame_limit=4e11,
    batch_size=24,
    data_warmup_steps=5000,
    input_length_lower_bound=0,
    input_length_upper_bound=160000,
    target_length_lower_bound=2,
    use_attention_mask=True,
    sample_partial_data=1.0,

    pad_token_id=0,

    reg_weight=5e-3,
    max_grad_norm=1.0,

    feat_proj_dropout=0.2,
    final_dropout=0.,
    hidden_dropout=0.2,
    activation_dropout=0.2,
    mask_time_prob=0.1,

    warmup_steps=5000,
    plateau_steps=0,
    reset_period=0,
    reset_times=1,
    max_lr=3e-4,
    min_lr=0.,
    final_lr=1.5e-6,
    decay_steps=75000,
    decay_rate=1e-2,
    decay_type='exp',
    adam_eps=1e-8,
    loss_normalize_type='sample',

    remove_unk=True,
    replace_apos=True,
    upper_only=False,

    asr_model_type='ctc',
    asr_vocab_size=32,
    asr_model_name='facebook/wav2vec2-large-lv60',
    asr_processor_name='',
    asr_keep_feature_encoder_only=False,
    asr_use_flash_attn=False,
    asr_use_mamba=False,
    asr_mamba_residual_inside=True,
    asr_mamba_use_mamba2=True,

    freeze_module='.',
    freeze_steps=10000,
    reinit_module='.',

    use_decoder=False,
    decoder_type='bart',
    decoder_name='facebook/bart-base',
    decoder_stack_encoder=False,
    decoder_model_name='',
    decoder_weight=1.0,
    ctc_weight=1.0,
    decoder_dropout=0.2,


    adaptor_type='conv',
    adaptor_num_layers=3,
    adaptor_pos_encoding=False,
    adaptor_pos_encoding_mode='encoding',
    adaptor_pe_weight=0.1,
    adaptor_use_layernorm=False,
    adaptor_use_glu=True,

    eval_num_beams=5,
    eval_length_penalty=1.0,

    teacher_snapshot_cycle=10000,
    alpha_layer_distill_loss=1,
    alpha_logits_distill_loss=1,

    use_classifier=False,
    classifier_num_targets="18,46",
    classifier_weight=1.0,
    classifier_pooling='mean',
)
