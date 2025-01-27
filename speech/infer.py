#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#

import os, logging, time, traceback
from utils import dict_send_to
import pickle
from models.model import compute_loss
import jiwer
import json
import torch
import argparse
import datetime
import sys
import numpy as np
from utils import infolog, checkpoint
from models import model, build
from dataloader import Feeder
from dataloader_hf import Feeder as FeederHF
from hyperparams import hparams as hp

metrics = {'wer': jiwer.wer, 'cer': jiwer.cer}

def infer_batches(model, batches, eval_path, hp, device='cpu', processor=None, write_output=True):
    if eval_path is not None:
        os.makedirs(eval_path, exist_ok=True)
    model.eval()
    if hasattr(model, 'module'):
        eval_model = model.module
    else:
        eval_model = model

    logging.info('Running %d evals, to %s' % (len(batches), eval_path))

    if processor:
        tokenizer = processor['tokenizer']
        decoder_tokenizer = processor['decoder_tokenizer']

    all_logits = []
    all_preds = []
    all_losses = []
    all_labels = []
    all_names = []
    all_decoder_preds = []
    all_clf_preds = []
    all_clf_labels = []
    all_time = []
    all_lengths = []

    n_samples = 0
    start_tic = time.time()
    for i, batch in enumerate(batches):
        try:
            eval_tic = time.time()
            batch_ = dict_send_to(batch, device)
            with torch.no_grad():
                outputs = eval_model.generate(**batch_,
                                              num_beams=hp.eval_num_beams,
                                              length_penalty=hp.eval_length_penalty)
            if 'loss' in outputs:
                losses = compute_loss(batch_, outputs, eval_model, hp)
                losses = dict_send_to(losses, 'cpu', detach=True, as_numpy=True)
            outputs = dict_send_to(outputs, 'cpu', detach=True, as_numpy=True)

            logits = outputs['logits']
            all_logits.extend(list(logits))
            n_samples += len(batch['names'])
            all_names.extend(batch['names'])
            if processor:
                preds = logits.argmax(axis=-1)
                preds = [tokenizer.decode(p) for p in preds]
                all_preds.extend(preds)

                if 'decoder_outputs' in outputs:
                    all_decoder_preds.extend(
                        decoder_tokenizer.batch_decode(outputs['decoder_outputs'], skip_special_tokens=True))

            if 'classifier_logits' in outputs:
                preds = torch.stack([t.argmax(axis=-1) for t in outputs['classifier_logits']]).T.detach().cpu().numpy()
                all_clf_preds.extend(preds.tolist())
                if 'classifier_labels' in batch:
                    all_clf_labels.extend(batch['classifier_labels'].numpy().tolist())

            if 'labels' in batch:
                if 'ctc_losses' in losses:
                    all_losses.extend(losses['ctc_losses'].tolist())
                labels = list(batch['labels'].numpy())
                for l in labels:
                    l[l == -100] = tokenizer.pad_token_id
                labels = [tokenizer.decode(l, group_tokens=False) for l in labels]
                all_labels.extend(labels)

            logging.info('Finished batch %d in %.2f sec, samples: %s' % (
                i, time.time() - eval_tic, batch['names']))
        except:
            traceback.print_exc()

    logging.info("Total %d batches of %d samples, cost %.2f sec" %
                 (len(batches), n_samples, time.time() - start_tic))
    return_metrics = {}
    if all_labels:
        all_loss = sum(all_losses) / len(all_losses) if all_losses else 0
        return_metrics['loss'] = all_loss
        if tokenizer:
            for key, fn in metrics.items():
                r = fn(all_labels, all_preds)
                logging.info("%s: %.4f" % (key, r))
                return_metrics[key] = r

                if all_decoder_preds:
                    r = fn(all_labels, all_decoder_preds)
                    logging.info("s2s_%s: %.4f" % (key, r))
                    return_metrics['decoder_' + key] = r
    else:
        all_loss = None

    if all_clf_labels:
        acc = (np.asarray(all_clf_preds) == np.asarray(all_clf_labels)).mean()
        logging.info("Classifier acc: %.4f" % acc)
        return_metrics['clf_acc'] = acc

    if write_output:
        fw = open(os.path.join(eval_path, 'preds.jsonl'), 'w')
        for i in range(len(all_preds)):
            r = {'pred': all_preds[i], 'name': all_names[i]}
            if all_labels:
                r['label'] = all_labels[i]
            if all_decoder_preds:
                r['decoder_pred'] = all_decoder_preds[i]
            if all_clf_preds:
                r['clf_pred'] = all_clf_preds[i]
            if all_clf_labels:
                r['clf_label'] = all_clf_labels[i]
            fw.write(json.dumps(r) + '\n')

        pickle.dump({'logits': all_logits, 'preds': all_preds, 'losses': all_losses,
                     'all_decoder_preds': all_decoder_preds,
                     'labels': all_labels, 'names': all_names, 'loss': all_loss,
                     'time': all_time, 'all_lengths': all_lengths},
                    open(os.path.join(eval_path, 'logits.pkl'), 'wb'))
    return return_metrics


def main(args):
    model_path = args.model_path
    if os.path.isdir(model_path):
        model_dir = model_path
    else:
        model_dir = os.path.dirname(model_path)
    logdir = args.output_path if args.output_path is not None else model_dir
    time_id = datetime.datetime.now().strftime('%m%d_%H%M')
    os.makedirs(logdir, exist_ok=True)
    data_dir = args.data_dir


    infolog.set_logger(os.path.join(logdir, 'outputs_%s.log' % (time_id)))
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
        keys_ = [k for k in keys if hasattr(hp, k)]
        hp_ = {k: v for k, v in hp_.items() if k in keys_}
        hp.override_from_dict(hp_)
    if args.hparams and os.path.isfile(args.hparams):
        hp.override_from_dict(json.load(open(args.hparams)))
    else:
        hp.parse(args.hparams)

    vocab_path = args.vocab_path if args.vocab_path else os.path.join(data_dir, 'vocab.json')
    processor = build.build_processor(hp, vocab_path)

    if args.use_hf_dataset:
        eval_split = args.eval_meta if args.eval_meta else 'validation'

        feeder_eval = FeederHF(data_dir, eval_split, processor, hparams=hp, filter_samples=False)
    else:
        eval_meta = args.eval_meta if args.eval_meta else os.path.join(data_dir, 'meta.dev.txt')
        if not os.path.exists(eval_meta):
            eval_meta = os.path.join(data_dir, args.eval_meta)
        feeder_eval = Feeder(data_dir, processor, eval_meta, hparams=hp, filter_samples=False)
    feeder_eval._batch_frame_limit /= 2  # Try to avoid OOM in eval
    feeder_eval._batch_quad_frame_limit /= 2
    feeder_eval.prepare_all_batches()

    m = model.Model(hp)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m.to(device)

    if not torch.cuda.is_available():
        map_location = lambda _, __: _.cpu()
    else:
        map_location = {}

    if not os.path.isdir(model_path):
        global_step = checkpoint.load_model(args.model_path, m, None, None, map_location)
        logging.info("Restore from " + args.model_path + ", step %s" % str(global_step))
    else:
        ckpt_path = checkpoint.find_ckpt(model_dir)
        if ckpt_path:
            global_step = checkpoint.load_model(ckpt_path, m, None, None, map_location)
            logging.info(
                "Restore from latest ckpt at " + model_dir + " from " + ckpt_path + ", step %s" % str(global_step))

    batches = feeder_eval.fetch_data()
    metrics = infer_batches(m, batches, logdir, hp, device, processor)
    json.dump(metrics, open(os.path.join(logdir, 'metrics.json'), 'w'), indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True,
                        help="Directory or path to restore model from")
    parser.add_argument('--output-path', required=True,
                        help="Directory or path to save results")
    parser.add_argument('--data-dir', help="Directory with data and metadata")
    parser.add_argument('--vocab-path', type=str, default=None, help="Path to vocab.json")
    parser.add_argument('--eval_meta', type=str, default=None,
                        help="Metadata file for eval, use metadata.eval.txt under data-dir when not given")
    parser.add_argument('--hparams', default='', help='Alternative hparams')
    parser.add_argument('--use_hf_dataset', action='store_true', help='Use Huggingface datasets')

    args, unparsed = parser.parse_known_args()
    print('unparsed:', unparsed)

    main(args)