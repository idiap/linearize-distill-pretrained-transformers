#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#

import torch
from torch import nn
from torch.nn import functional as F
from models.build import build_asr_model, build_text_decoder, Adaptor, Classifier
from transformers.utils import ModelOutput
import logging

class Model(nn.Module):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.hparams = hparams
        self.asr_model = build_asr_model(hparams)
        if hparams.use_decoder:
            self.decoder = build_text_decoder(hparams)
            self.adaptor = Adaptor(hparams, self.asr_model.config.hidden_size, self.decoder.config.d_model)
        if hparams.use_classifier:
            self.classifier = Classifier(self.asr_model.config.hidden_size,
                                         [int(t) for t in hparams.classifier_num_targets.split(',')],
                                         pooling_type=hparams.classifier_pooling)


    def get_output_length(self, input_masks=None, input_lengths=None):
        if input_masks is not None:
            return self.asr_model._get_feat_extract_output_lengths(
                input_masks.sum(-1)
            )
        elif input_lengths is not None:
            return self.asr_model._get_feat_extract_output_lengths(
                input_lengths
            )

    def forward(self, inputs, input_masks=None, labels=None, decoder_labels=None, decoder_label_masks=None, **kwargs):
        outputs = self.asr_model(input_values=inputs, attention_mask=input_masks, labels=labels,
                                 output_hidden_states=True, return_dict=True)

        mask = self.asr_model._get_feature_vector_attention_mask(
            outputs['logits'].shape[1], input_masks, add_adapter=False
        )
        outputs['output_masks'] = mask
        if self.hparams.use_decoder:

            hidden = self.adaptor(outputs['hidden_states'][-1])
            mask = self.adaptor.transform_attention_mask(mask)
            if self.hparams.decoder_stack_encoder:
                args = {'inputs_embeds': hidden}
            else:
                args = {'encoder_outputs': (hidden,)}
            decoder_outputs = self.decoder(attention_mask=mask,
                                           labels=decoder_labels,
                                           decoder_attention_mask=decoder_label_masks,
                                           output_hidden_states=True,
                                           return_dict=True, **args)
            outputs['decoder_outputs'] = decoder_outputs
        if self.hparams.use_classifier:
            logits = self.classifier(outputs['hidden_states'][-1], mask)
            outputs['classifier_logits'] = logits
        return outputs

    def generate(self, inputs, input_masks=None, labels=None, **kwargs):
        outputs = self.asr_model(input_values=inputs, attention_mask=input_masks, labels=labels,
                                 output_hidden_states=True, return_dict=True)
        mask = self.asr_model._get_feature_vector_attention_mask(
            outputs['logits'].shape[1], input_masks, add_adapter=False
        )
        outputs['output_masks'] = mask
        if self.hparams.use_decoder:
            mask = self.asr_model._get_feature_vector_attention_mask(
                outputs['logits'].shape[1], input_masks, add_adapter=False
            )

            hidden = self.adaptor(outputs['hidden_states'][-1])
            mask = self.adaptor.transform_attention_mask(mask)
            if self.hparams.decoder_stack_encoder:
                kwargs['inputs_embeds'] = hidden
                for key in list(kwargs.keys()):
                    if key in ['input_lengths', 'names', 'label_lengths', 'label_masks'] or key.startswith('decoder_'):
                        del kwargs[key]
            else:
                kwargs['encoder_outputs'] = ModelOutput(last_hidden_state=hidden)
            decoder_outputs = self.decoder.generate(output_attentions=True, attention_mask=mask,
                                                    **kwargs)
            outputs['decoder_outputs'] = decoder_outputs
        if self.hparams.use_classifier:
            logits = self.classifier(outputs['hidden_states'][-1], mask)
            outputs['classifier_logits'] = logits
        return outputs


def learning_rate_schedule(global_step, hp):
    if hp.reset_period > 0:
        n_reset_times = global_step // hp.reset_period
        n_reset_times = min(n_reset_times, hp.reset_times)
        global_step -= n_reset_times * hp.reset_period
    if global_step < hp.warmup_steps:
        return ((hp.max_lr - hp.min_lr) * (global_step / hp.warmup_steps) + hp.min_lr) / hp.max_lr
    elif global_step <= hp.warmup_steps + hp.plateau_steps:
        return 1
    elif global_step < hp.warmup_steps + hp.plateau_steps + hp.decay_steps:
        if hp.decay_type == 'exp':
            decay_factor = -torch.log(torch.tensor(hp.final_lr / hp.max_lr)) / hp.decay_steps
            return torch.exp(- (global_step - hp.warmup_steps - hp.plateau_steps) * decay_factor)
        elif hp.decay_type == 'linear':
            decay_factor = (hp.max_lr - hp.final_lr) / hp.decay_steps
            return 1 - (global_step - hp.warmup_steps - hp.plateau_steps) * decay_factor / hp.max_lr
        else:
            raise ValueError('Unknown decay type: %s' % hp.decay_type)
    else:
        return hp.final_lr / hp.max_lr


def is_weight_decayed(n):
    return n.split('.')[-1] != 'bias' and n.split('.')[-2] != 'layer_norm'

import torch.nn.functional as F
def layer_distill_loss(all_teacher_features, all_student_features, mask, normalize_type,
                       distill_normalization='layernorm'):
    layer_distill_losses = []
    lengths = mask.sum(-1).maximum(torch.tensor(1, device=mask.device))
    mask = mask.unsqueeze(-1)
    # base_distill_losses = []
    for i in range(1, len(all_teacher_features)):
        teacher_features = all_teacher_features[i]
        student_features = all_student_features[i]
        if mask is not None:
            teacher_features = teacher_features * mask
            student_features = student_features * mask
        # base_distill_losses.append(t_layer_distill_loss.detach())
        if distill_normalization == 'layernorm':
            teacher_features = torch.layer_norm(teacher_features, [teacher_features.shape[-1]])
            student_features = torch.layer_norm(student_features, [student_features.shape[-1]])
            t_layer_distill_loss = F.mse_loss(teacher_features, student_features, reduction='none').mean(-1)
        else:
            t_layer_distill_loss = F.mse_loss(teacher_features, student_features, reduction='none').mean(-1)
        if normalize_type == 'sample':
            t_layer_distill_loss = t_layer_distill_loss.sum()
        else:
            t_layer_distill_loss = (t_layer_distill_loss.sum(-1) / lengths).sum()
        layer_distill_losses.append(t_layer_distill_loss)
    layer_loss = torch.stack(layer_distill_losses).mean()
    return layer_loss
    # return torch.stack(base_distill_losses).mean()#, layer_loss



def compute_loss(batch, outputs, model, hp):
    if hp.ctc_weight > 0 and hp.use_decoder and hp.decoder_weight > 0 and hp.loss_normalize_type == 'samples':
        raise ValueError('Cannot normalize loss by samples when using both decoder and CTC')

    batch_size = batch['inputs'].size(0)
    finite_mask = outputs['loss'] != 0  # inf has been masked out
    n_finite = finite_mask.sum()
    if hp.ctc_weight == 0.:
        finite_mask = torch.ones_like(finite_mask)
    n_frames = (finite_mask * batch['label_lengths']).sum()
    if hp.loss_normalize_type == 'utterance':
        losses = outputs['loss'] / batch['label_lengths'].float()
        ctc_loss = losses.sum()
        n_samples = n_finite
    elif hp.loss_normalize_type == 'sample':
        losses = outputs['loss'].clone()
        ctc_loss = outputs['loss'].sum()
        n_samples = n_frames
    else:
        raise ValueError('Unknown loss normalization type: %s' % hp.loss_normalize_type)

    result = {}
    if hp.ctc_weight > 0:
        loss = ctc_loss * hp.ctc_weight
        result['ctc_loss'] = ctc_loss
        result['ctc_losses'] = losses
    else:
        loss = 0.

    if hp.use_decoder and isinstance(outputs['decoder_outputs'], dict):
        logits = outputs['decoder_outputs']['logits']
        decoder_losses = F.cross_entropy(logits.transpose(1, 2), batch['decoder_labels'], reduction='none')
        if hp.loss_normalize_type == 'sample':
            n_samples = batch['decoder_label_lengths'].sum()
            decoder_loss = decoder_losses.sum()
        elif hp.loss_normalize_type == 'utterance':
            n_samples = n_finite
            decoder_losses = decoder_losses.sum(-1) / batch['decoder_label_lengths'].float()
            decoder_loss = decoder_losses.sum()
        loss = loss + decoder_loss * hp.decoder_weight
        result['decoder_losses'] = decoder_losses
        result['decoder_loss'] = decoder_loss

    if 'teacher_outputs' in outputs:
        teacher_outputs = outputs['teacher_outputs']
        mask = outputs['output_masks'] * finite_mask.unsqueeze(-1)
        if hp.loss_normalize_type == 'utterance':
            scale_factor = 1
        else:
            scale_factor = n_samples / outputs['total_output_samples']
        if hp.alpha_layer_distill_loss > 0:
            layer_loss = layer_distill_loss(teacher_outputs['hidden_states'], outputs['hidden_states'],
                                            mask, hp.loss_normalize_type)
            layer_loss = layer_loss * scale_factor
            loss = loss + hp.alpha_layer_distill_loss * layer_loss
            result['layer_loss'] = layer_loss

        if hp.alpha_logits_distill_loss > 0:
            temparature = 2.0
            logits_distill_losses = F.kl_div(
                input=F.log_softmax(outputs['logits'] / temparature, dim=-1),
                target=F.softmax(teacher_outputs['logits'] / temparature, dim=-1),
                reduction="none"
            ).sum(-1)
            if hp.loss_normalize_type == 'sample':
                logits_distill_loss = (logits_distill_losses * mask).sum()
            else:
                lengths = mask.sum(-1).maximum(torch.tensor(1, device=mask.device))
                logits_distill_loss = (logits_distill_losses.sum(-1) / lengths).sum()
            logits_distill_loss = logits_distill_loss * scale_factor
            loss = loss + hp.alpha_logits_distill_loss * logits_distill_loss
            result['logit_loss'] = logits_distill_loss

    if 'classifier_logits' in outputs:
        logits = outputs['classifier_logits']
        classifier_loss = 0.
        for i in range(len(logits)):
            cls_loss = F.cross_entropy(logits[i], batch['classifier_labels'][:, i], reduction='none')
            classifier_loss += cls_loss.sum()
        classifier_loss = classifier_loss / len(logits)
        loss = loss + classifier_loss * hp.classifier_weight
        result['classifier_loss'] = classifier_loss

    result.update({'loss': loss,
                   'n_inf': batch_size - n_finite, 'n_fin': n_finite, 'batch_size': batch_size, 'n_frames': n_frames,
                   'n_samples': n_samples})
    return result

def freeze_module(model, prefix, frozen=True):
    if hasattr(model, 'module'):
        model = model.module

    names = []

    for name, param in model.named_parameters():
        if name.startswith(prefix):
            param.requires_grad = not frozen
            names.append(name)
    if names:
        logging.info("%s parameters of prefix %s:" % ('Freeze' if frozen else 'Defreeze', prefix))
        logging.info(", ".join(names))
    model.asr_model.freeze_feature_encoder()

def init_module(model, prefix):
    if hasattr(model, 'module'):
        model = model.module
    base_module = [('asr_model', model.asr_model)]
    if model.hparams.use_decoder:
        base_module.append(('decoder', model.decoder))
    for bn, m in base_module:
        for name, module in m.named_modules():
            name = bn + '.' + name
            if name.startswith(prefix):
                m._init_weights(module)