#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#

import gc
import pickle
from collections import defaultdict

import os, glob
import traceback as tb

from utils import dict_send_to
from tqdm import tqdm
import time, datetime
import argparse
import json
import traceback
from hyperparams import hparams as hp
import torch
from torch import nn
import logging
from utils import infolog, checkpoint
from models import model, build
from dataloader import Feeder
from dataloader_hf import Feeder as FeederHF
from functools import partial
import sys
import faulthandler, signal
from datetime import timedelta
from infer import infer_batches
from accelerate import Accelerator

if hasattr(faulthandler, 'register'):
    faulthandler.register(signal.SIGUSR1)


def main(args):
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model_dir = args.model_dir
    logdir = args.log_dir if args.log_dir is not None else model_dir
    data_dir = args.data_dir
    run_name = os.path.split(logdir)[-1]

    if args.rewrite_output_dir and os.path.exists(model_dir):
        logging.info('Removing existing model dir %s' % model_dir)
        os.system('rm -rf %s' % model_dir)

    if os.path.exists(model_dir) and os.listdir(model_dir) and args.restore_from is None:
        args.restore_from = model_dir

    if args.restore_from and os.path.isdir(args.restore_from):
        logdir = args.restore_from
        model_dir = args.restore_from
        args.restore_from = None

    time_id = datetime.datetime.now().strftime('%m%d_%H%M')

    torch.manual_seed(0)
    if args.ddp:
        from torch import distributed as dist
        dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=10))
        rank = dist.get_rank()
        local_rank = args.local_rank
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        map_location = lambda _, __: _.cuda(local_rank)
        print("Rank: %d, Local rank: %d, World size: %d" % (rank, local_rank, world_size))
    else:
        rank = local_rank = 0
        world_size = 1
        map_location = {}

    accelerator = Accelerator(mixed_precision=args.mixed_precision,
                              log_with=['wandb'],
                              project_dir=logdir, )
    if rank == 0 and not args.no_write:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(logdir, exist_ok=True)

        keep_log = True
        experiment_config = vars(args)
        experiment_config.update(vars(hp))
        del experiment_config['_hparam_types']
        kwargs = {"wandb": {'name': run_name}}
        if args.resume_run_id:
            kwargs['wandb']["id"] = args.resume_run_id
            kwargs['wandb']["resume"] = "must"
        accelerator.init_trackers(args.project_name,
                                  experiment_config,
                                  init_kwargs=kwargs)
    else:
        keep_log = False

    infolog.set_logger(os.path.join(logdir, 'outputs_%s_%d.log' % (time_id, rank)) if not args.no_write else None)
    sys.stdout = infolog.StreamToLogger(logging.root, logging.INFO)
    sys.stderr = infolog.StreamToLogger(logging.root, logging.ERROR)

    logging.info("Command: " + str(' '.join(sys.argv)))
    if os.path.exists(os.path.join(model_dir, 'hparams.json')):
        hp_ = json.load(open(os.path.join(model_dir, 'hparams.json')))
        keys = set(hp_.keys()).union(hp._hparam_types.keys())
        logging.info("Restoring hparams...")
        for k in keys:
            if hp.get(k, None) != hp_.get(k, None):
                logging.info("Different hparam %s: %s -> %s" % (k, str(hp.get(k, None)), str(hp_.get(k, None))))
        hp.override_from_dict(hp_)
    if args.hparams and os.path.isfile(args.hparams):
        hp.override_from_dict(json.load(open(args.hparams)))
    else:
        hp.parse(args.hparams)

    if os.path.exists(os.path.join(model_dir, 'args.json')):
        args_ = json.load(open(os.path.join(model_dir, 'args.json')))
        args_c = vars(args)
        keys = set(args_.keys()).union(args_c.keys())
        logging.info("Found args from previous run...")
        for k in keys:
            if args_.get(k, '') != args_c.get(k, ''):
                logging.info("Changed arg %s: %s -> %s" % (k, str(args_.get(k, '')), str(args_c.get(k, ''))))

    if os.path.exists(os.path.join(model_dir, 'best_metrics.json')):
        best_metrics = json.load(open(os.path.join(model_dir, 'best_metrics.json')))
    else:
        best_metrics = {}

    if not torch.cuda.is_available():
        map_location = lambda _, __: _.cpu()

    if rank == 0:
        values = hp.values()
        logging.info('Hyperparameters:\n' + '\n'.join(['  %s: %s' % (name, values[name]) for name in sorted(values)]))

        if not args.no_write:
            if os.path.exists(os.path.join(model_dir, 'hparams.json')):
                os.rename(os.path.join(model_dir, 'hparams.json'),
                          os.path.join(model_dir, 'hparams.json.' + time_id))
            if os.path.exists(os.path.join(model_dir, 'args.json')):
                os.rename(os.path.join(model_dir, 'args.json'),
                          os.path.join(model_dir, 'args.json.' + time_id))
            open(os.path.join(logdir, 'hparams.json'), 'w').write(hp.to_json(indent=1))
            open(os.path.join(logdir, 'args.json'), 'w').write(json.dumps(vars(args), indent=1))

    if args.eval_steps is not None:
        eval_steps = [int(s) for s in args.eval_steps.split(':')]
    else:
        eval_steps = None

    vocab_path = args.vocab_path if args.vocab_path else os.path.join(data_dir, 'vocab.json')

    processor = build.build_processor(hp, vocab_path)

    if args.use_hf_dataset:
        train_split = args.train_meta if args.train_meta else 'train'
        eval_split = args.eval_meta if args.eval_meta else 'validation'
        feeder = FeederHF(data_dir, train_split, processor, hparams=hp,
                          rank=rank, world_size=world_size, shuffle=hp.shuffle_training_data,
                          sample_partial=hp.sample_partial_data)
        if rank == 0:
            feeder_eval = FeederHF(data_dir, eval_split, processor, hparams=hp, filter_samples=False)
    else:
        train_meta = args.train_meta if args.train_meta else os.path.join(data_dir, 'meta.train.txt')
        eval_meta = args.eval_meta if args.eval_meta else os.path.join(data_dir, 'meta.dev.txt')
        feeder = Feeder(data_dir, processor, train_meta, hparams=hp,
                        rank=rank, world_size=world_size, shuffle=hp.shuffle_training_data,
                        sample_partial=hp.sample_partial_data)
        if rank == 0:
            feeder_eval = Feeder(data_dir, processor, eval_meta, hparams=hp, filter_samples=False)

    if rank == 0:
        feeder_eval._batch_frame_limit /= 2 # Try to avoid OOM in eval
        feeder_eval._batch_quad_frame_limit /= 2
        feeder_eval.prepare_all_batches()

    logging.info("Using %d GPUs" % torch.cuda.device_count())
    m = model.Model(hp)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m.to(device)
    if args.ddp:
        example_param = list(m.parameters())[5]
        logging.info("Model on %s" % str(example_param.device))
        m = nn.parallel.DistributedDataParallel(m, device_ids=[local_rank], output_device=local_rank)
    elif torch.cuda.device_count() > 1:
        m = nn.DataParallel(m)
    else:
        from models.wrapper import DummyWrapper
        m = DummyWrapper(m)

    wd, nwd = [], []
    for name, param in m.named_parameters():
        if model.is_weight_decayed(name):
            wd.append(param)
        else:
            nwd.append(param)

    optim = torch.optim.AdamW([{'params': wd, 'weight_decay': hp.reg_weight}, {'params': nwd, 'weight_decay': 0.}],
                              lr=hp.max_lr, eps=hp.adam_eps, betas=(0.9, 0.999))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=partial(model.learning_rate_schedule, hp=hp))

    global_step = None
    if args.restore_from:
        global_step = checkpoint.load_model(args.restore_from, m, optim, sched, map_location, args.reset_training)
        logging.info("Restore from" + args.restore_from + ", step %s" % str(global_step))
    ckpt_path = checkpoint.find_ckpt(model_dir)
    if ckpt_path:
        global_step = checkpoint.load_model(ckpt_path, m, optim, sched, map_location)
        logging.info("Restore from previous run at " + model_dir + " from " + ckpt_path + ", step %s" % str(global_step))
    if global_step is None:
        global_step = 0
    if os.path.exists(os.path.join(logdir, 'feeder_%d.pth' % rank)):
        feeder.load_state_dict(torch.load(os.path.join(logdir, 'feeder_%d.pth' % rank)))

    feeder.global_step = global_step
    feeder.daemon = True
    feeder.start()
    m.train()

    time_window = infolog.ValueWindow(100)
    loss_window = infolog.ValueWindow(100)
    recent_fails = infolog.ValueWindow(10)
    summary_windows = []
    if rank == 0:
        state_dict = m.state_dict()
        for var in state_dict:
            logging.info("%s %s" % (var, state_dict[var].shape))

    if global_step < hp.freeze_steps:
        model.freeze_module(m, hp.freeze_module)
    model.init_module(m, hp.reinit_module)

    logging.info("Start training run")
    optim.zero_grad()
    accum = args.accumulation_steps

    m, optim, sched = accelerator.prepare(m, optim, sched)

    length_window = infolog.ValueWindow(100)
    sample_window = infolog.ValueWindow(100)

    def signal_handler(sig, frame):
        if not args.no_write:
            logging.info("Got signal %d, saving and exiting" % sig)
            checkpoint.save_model(model_dir, m, optim, sched, global_step)
            logging.info("Save checkpoint to " + model_dir)
            torch.save(feeder.state_dict(), os.path.join(logdir, 'feeder_%d_%d.pth' % (global_step, rank)))
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    while True:
        tic = time.time()
        accum_losses = defaultdict(float)
        accum_loss_keys = ['loss', 'ctc_loss', 'decoder_loss', 'classifier_loss']
        accum_keys = accum_loss_keys + ['n_inf', 'n_fin', 'n_frames', 'n_samples']


        accum_batches = [feeder.get_batch() for _ in range(accum)]
        output_lengths = [m.module.get_output_length(batch['input_masks']) for batch in accum_batches]
        total_output_samples = sum([l.sum() for l in output_lengths])



        for i, batch in enumerate(accum_batches):
            # logging.info("%s %.2E %.2E" % (str(batch['inputs'].shape), batch['inputs'].shape[0] * batch['inputs'].shape[1],
            #                            batch['inputs'].shape[0] * batch['inputs'].shape[1] * batch['inputs'].shape[1]))
            # logging.info("%s (%.2f) %d (%.2f) %d" % (str(batch['inputs'].shape), sample_window.average,
            #                                          batch['input_lengths'].sum(), length_window.average,
            #                                          batch['label_lengths'].sum(), label_length_window.average))
            # length_window.append(batch['input_lengths'].sum())
            # label_length_window.append(batch['label_lengths'].sum())
            # sample_window.append(batch['inputs'].shape[0])
            batch = dict_send_to(batch, device, non_blocking=True)
            oom = False
            try:
                outputs = m(**batch)
                outputs['total_output_samples'] = total_output_samples
                with accelerator.autocast():
                    losses = model.compute_loss(batch, outputs, m, hp)
                accelerator.backward(losses['loss'])

                if 'ctc_losses' in losses:
                    losses.pop('ctc_losses')
                losses = dict_send_to(losses, torch.device('cpu'), detach=True)
                for key in accum_keys:
                    if key in losses:
                        accum_losses[key] += losses[key]

                del outputs
                del losses
                del batch
            except Exception as e:
                logging.error("Failed due to %s, input shape: %s, target shape: %s" %
                              (type(e).__name__, str(batch['inputs'].shape), str(batch['labels'].shape)))
                traceback.print_exc()
                if len(recent_fails._values) == 10 and recent_fails._values[0] > global_step - 20:
                    logging.error("Too many failures, exiting")
                    return
                recent_fails.append(global_step)
                optim.zero_grad()
                oom = True

            if oom:
                gc.collect()
                if args.ddp or torch.cuda.memory_allocated() > torch.cuda.max_memory_allocated() * 0.9:
                    logging.info("Current memory: %d, cf. peak %d" %
                                 (torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()))
                    if not args.no_write:
                        torch.save(feeder.state_dict(), os.path.join(logdir, 'feeder_%d.pth' % rank))
                        if rank == 0:
                            checkpoint.save_model(model_dir, m, optim, sched, global_step)
                        else:
                            time.sleep(20)
                    sys.exit(1)
                continue

        if accum_losses['n_samples'] > 1:
            for p in m.parameters():
                if p.grad is not None:
                    p.grad /= accum_losses['n_samples']
        elif accum_losses['n_samples'] == 0:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), hp.max_grad_norm)
        optim.step()
        optim.zero_grad()
        sched.step()
        global_step += 1
        feeder.global_step = global_step

        if global_step == hp.freeze_steps:
            model.freeze_module(m, hp.freeze_module, frozen=False)

        if rank == 0:
            losses = {}
            for key in accum_losses:
                if key in accum_loss_keys:
                    losses[key] = accum_losses[key] / accum_losses['n_samples']
                else:
                    losses[key] = accum_losses[key]

            grad_norm_value = grad_norm.to('cpu').item()
            dur = time.time() - tic
            time_window.append(dur)
            loss_window.append(losses['loss'])
            sample_window.append(losses['n_fin'])
            length_window.append(losses['n_frames'])
            loss_message = ', '.join(['%s=%.4f' % (k, losses[k]) for k in accum_loss_keys if k in losses])
            message = '[Step %d] %.3f sec/step (%.3f), lr=%.04E, %s (Ave. loss %.5f), %d (%.2f) samples, %d (%.2f) frames, grad_norm=%.3f' % (
                global_step, dur, time_window.average, sched.get_last_lr()[-1], loss_message,
                loss_window.average, losses['n_fin'], sample_window.average,
                losses['n_frames'], length_window.average, grad_norm_value)
            if losses['n_inf']:
                message += ', %d infinite [WARN]' % (losses['n_inf'])
            logging.info(message)

            if global_step % args.checkpoint_interval == 0 and not args.no_write:
                checkpoint.save_model(model_dir, m, optim, sched, global_step)
                logging.info("Save checkpoint to " + model_dir)
                checkpoint.cleanup_checkpoint(model_dir, [args.checkpoint_interval, args.eval_interval])

            if global_step % args.summary_interval == 0 and keep_log:
                log_dict = {'lr': sched.get_last_lr()[-1], 'grad_norm': grad_norm}
                for key in accum_keys:
                    if key in losses:
                        log_dict['losses/' + key] = losses[key]

                for window in summary_windows:
                    stats = window.summary()
                    for k, v in stats:
                        log_dict[k] = v
                accelerator.log(log_dict, step=global_step)

            if (eval_steps and global_step in eval_steps) or \
                    (eval_steps is None and
                     ((global_step % args.checkpoint_interval == 0) or
                      (global_step % args.eval_interval == 0))):
                eval_path = os.path.join(logdir, 'eval_%d' % (global_step))
                batches = feeder_eval.fetch_data()
                batches = batches[:hp.max_eval_batches]

                with accelerator.autocast():
                    metrics = infer_batches(m, batches, eval_path, hp, device, processor, write_output=not args.no_write)

                if not args.no_write:
                    smaller_better = ['wer', 'cer']
                    updated_bests = []
                    log_dict = {}
                    for k, v in metrics.items():
                        log_dict['eval/%s' % k] = v
                        if k in smaller_better and metrics[k] < best_metrics.get(k, (float('inf'), None))[0]:
                            updated_bests.append(k)
                        elif k not in smaller_better and metrics[k] > best_metrics.get(k, (float('-inf'), None))[0]:
                            updated_bests.append(k)
                    accelerator.log(log_dict, step=global_step)
                    if updated_bests:
                        name = 'model.ckpt-%d' % global_step
                        for k in updated_bests:
                            name += '-%s-%.4f' % (k, metrics[k])
                            best_metrics[k] = (metrics[k], global_step)

                        json.dump(best_metrics, open(os.path.join(logdir, 'best_metrics.json'), 'w'))

                        if global_step >= args.checkpoint_interval:
                            checkpoint.save_model(model_dir, m, optim, sched, global_step, name=name)
                            logging.info("Save best model to " + name)

                            if os.path.exists(os.path.join(model_dir, 'model.ckpt-%d' % global_step)):
                                os.remove(os.path.join(model_dir, 'model.ckpt-%d' % global_step))

                            checkpoint.cleanup_checkpoint(model_dir, [args.checkpoint_interval, args.eval_interval])


                m.train()
        if args.max_steps is not None and global_step >= args.max_steps:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True,
                        help="Directory to save checkpoints and resume (when --restore_from is not specified)")
    parser.add_argument('--log-dir', default=None, help="Directory to save log and tfevents")
    parser.add_argument('--data-dir', required=True, help="Directory with data and metadata")
    parser.add_argument('--use_hf_dataset', action='store_true', help="Use Huggingface dataset")
    parser.add_argument('--vocab-path', type=str, default=None, help="Path to vocab.json")
    parser.add_argument('--train_meta', type=str, default=None,
                        help="Metadata file for training, use metadata.train.txt under data-dir when not given")
    parser.add_argument('--eval_meta', type=str, default=None,
                        help="Metadata file for eval, use metadata.eval.txt under data-dir when not given")
    parser.add_argument('--eval_steps', type=str, default=None,
                        help="Steps of checkpoints to run eval on. Run on all steps when not specified")
    parser.add_argument('--checkpoint_interval', type=int, default=5000)
    parser.add_argument('--eval_interval', type=int, default=2000)
    parser.add_argument('--summary_interval', type=int, default=20)
    parser.add_argument('--restore_from', help='Path of checkpoint or run directory to restore', default=None)
    parser.add_argument('--hparams', default='', help='Alternative hparams')
    parser.add_argument('--ddp', help='Using DDP', action='store_true')
    parser.add_argument('--mixed_precision', help='Using mixed precision', type=str, default=None)
    parser.add_argument('--use_tf32', help='Using tf32', action='store_true')
    parser.add_argument('--max_retry', help='Number of max retry', type=int, default=0)
    parser.add_argument('--accumulation_steps', help='Number of steps for gradient accumulation', type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--rewrite_output_dir", action='store_true', default=False)
    parser.add_argument("--no_write", action='store_true', help="Prevent from writing any files", default=False)
    parser.add_argument("--reset_training", action='store_true', default=False)
    parser.add_argument("--resume_run_id", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument('--project_name', type=str, default='speech_uptrain')

    args, unparsed = parser.parse_known_args()
    print('unparsed:', unparsed)
    if args.max_retry == 0:
        try:
            main(args)
        except:
            tb.print_exc()
    else:
        import multiprocessing as mp
        from multiprocessing import Process

        mp.set_start_method('spawn')
        for i in range(args.max_retry + 1):
            if i != 0:
                print("\n==========Retry %d==========\n" % i)
            p = Process(target=main, args=(args,))
            p.start()
            p.join()
            if p.exitcode == 0:
                print("Success")
                i = None
                break
        if i is not None:
            print("Max retry reached...")
