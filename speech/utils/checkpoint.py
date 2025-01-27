#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#

import os
from collections import OrderedDict
import torch
import glob
import logging


def find_ckpt(base_dir):
    max_step = 0
    result = None
    for f in glob.iglob(os.path.join(base_dir, 'model.ckpt-*')):
        step = int(os.path.split(f)[-1].split('-')[1])
        if step > max_step:
            result = f
            max_step = step
    return result

def find_all_ckpt(base_dir):
    result = []
    for f in glob.iglob(os.path.join(base_dir, 'model.ckpt-*')):
        step = int(os.path.split(f)[-1].split('-')[1])
        result.append((step, f))
    return result

def cleanup_checkpoint(base_dir, intervals=[10]):
    for f in glob.iglob(os.path.join(base_dir, 'model.ckpt-*')):
        step = int(os.path.split(f)[-1].split('-')[1])
        if all(step % i != 0 for i in intervals):
            logging.info("Remove checkpoint %s" % f)
            os.remove(f)


def save_model(model_dir, model=None, optim=None, sched=None, step=None, name=None):
    state_dict = {}
    if model:
        if hasattr(model, 'module'):
            model_dict = model.module.state_dict()
        else:
            model_dict = model.state_dict()
        state_dict['model'] = model_dict
    if optim:
        state_dict['optim'] = optim.state_dict()
    if sched:
        state_dict['sched'] = sched.state_dict()
    if step:
        state_dict['step'] = step
        model_dir = os.path.join(model_dir, name if name else 'model.ckpt-%d' % step)
    torch.save(state_dict, model_dir)


def load_model(model_path, model=None, optim=None, sched=None, map_location={}, restart=False, load_no_optim=False):
    state_dict = torch.load(model_path, map_location=map_location)
    if 'model' in state_dict and model:
        model_dict = state_dict['model']
        if hasattr(model, 'module'):
            model = model.module
        if set(model.state_dict().keys()) != set(model_dict.keys()) and restart:
            logging.warning('Model parameters do not match, loading from checkpoint anyway')
            logging.warning("Missing parameters: %s" % (set(model.state_dict().keys()) - set(model_dict.keys())))
            logging.warning("Extra parameters: %s" % (set(model_dict.keys()) - set(model.state_dict().keys())))

        model.load_state_dict(model_dict, strict=not restart)
    if 'optim' in state_dict and optim and not restart and not load_no_optim:
        optim.load_state_dict(state_dict['optim'])
    if 'step' in state_dict and not restart:
        step = state_dict['step']
    else:
        step = None
    if 'sched' in state_dict and sched and not restart:
        sched.load_state_dict(state_dict['sched'])
        if step:
            if step != sched.last_epoch:
                logging.warn("Step=%d, while in sched step=%d" % (step, sched.last_epoch))
        else:
            step = sched.last_epoch
    return step
