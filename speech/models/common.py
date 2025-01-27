#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#

import numpy as np
import torch

def impute(x, lengths, channels_last=True):
    """ Set elements of a batch of a sequence of tensors to zero according to sequence lengths.
    :param x: A tensor with shape [batch, time_step, ...] or [batch, ..., time_step]
    :param lengths: A tensor with shape [batch]
    :param channels_last: A bool. If true, the time_step dimension is the second dimension, otherwise the last.

    :returns: A tensor with the same shape of x, with elements time_step > corresponding length set to 0.
    """

    if channels_last:
        max_length = x.shape[1]
    else:
        max_length = x.shape[-1]
    mask = torch.arange(max_length, device=lengths.device)[None, :] < lengths[:, None]  # [B, T]
    for _ in range(len(x.shape) - 2):
        if channels_last:
            mask = mask.unsqueeze(-1)
        else:
            mask = mask.unsqueeze(1)
    return x * mask


def mask_reduce(loss, lengths, per_sample=False):
    """ Reduce a batch of sequences according to the lengths of each sequence
    :param loss: A tensor with shape [batch, time_step]
    :param lengths: A tensor with shape [batch]
    :param per_sample: A bool.

    :returns: If per_sample, return a tensor with shape [batch], the loss averaged over the valid elements on each
    sequence; otherwise, return a scalar, the loss averaged over the entire batch.
    """

    if per_sample:
        loss = impute(loss, lengths).sum(-1) / lengths  # [B]
    else:
        loss = impute(loss, lengths).sum() / lengths.sum()
    return loss
