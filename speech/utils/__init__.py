#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#

import torch

def dict_send_to(data, device=None, detach=False, as_numpy=False, non_blocking=False):
    result = {}
    for key in data:
        t = data[key]
        if isinstance(t, torch.Tensor):
            if detach:
                t = t.detach()
            if device is not None:
                t = t.to(device)
            if as_numpy:
                if t.dtype == torch.bfloat16:
                    t = t.float()
                t = t.numpy()
        elif isinstance(t, dict):
            t = dict_send_to(t, device, detach, as_numpy)
        result[key] = t
    return result