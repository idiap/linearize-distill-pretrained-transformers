#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#

import io
import logging
import threading
import queue
import traceback
from collections import defaultdict
import time
import os
import json
import pickle
import zipfile

import soundfile as sd
import numpy as np
import torch

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer

np.random.seed(0)


class Feeder(threading.Thread):
    def __init__(self, datadir, processor, metadata_file_path, hparams,
                 rank=0, world_size=1, shuffle=True, prepare_all=False, single=False, filter_samples=True,
                 sample_partial=1):
        super(Feeder, self).__init__()
        self._offset = 0
        self._epoch = 0
        self._hparams = hparams
        self.global_step = -1
        self.proto = get_input_proto(hparams)
        self.queue = queue.Queue(maxsize=64)
        self.rand = np.random.RandomState(rank)
        self._rank = rank
        self._world_size = world_size
        self._lock = threading.Lock()

        self._datadir = datadir
        self._tokenizer = processor['tokenizer']
        self._extractor = processor['extractor']
        self._decoder_tokenizer = processor['decoder_tokenizer']

        self._batch_size = hparams.batch_size
        self._batch_frame_limit = hparams.batch_frame_limit
        self._batch_quad_frame_limit = hparams.batch_quad_frame_limit

        self._shuffle = shuffle
        self._single = single
        self._filter_samples = filter_samples

        # Load metadata
        with open(metadata_file_path, encoding='utf-8') as f:
            self._metadata = _read_meta(f, self._hparams.data_format)
        logging.info('%d samples read' % (len(self._metadata)))
        if sample_partial != 1:
            if sample_partial < 1:
                sample_partial = int(len(self._metadata) * sample_partial)
            self._metadata = self._metadata[:sample_partial]
            logging.info('Sampled %d examples' % len(self._metadata))

        hours = sum([int(x['l']) for x in self._metadata]) / hparams.sr / 3600
        logging.info('Loaded metadata for %d examples (%.2f hours)' % (len(self._metadata), hours))

        if hparams.use_classifier:
            self._cls_vocab = json.load(open(os.path.join(datadir, 'categories.json'), 'r'))
            if hparams.classifier_num_targets == 1 and isinstance(self._cls_vocab, dict):
                self._cls_vocab = [self._cls_vocab]
            for dim_i in range(len(self._cls_vocab)):
                if isinstance(self._cls_vocab[dim_i], int):
                    self._cls_vocab[dim_i] = list(range(self._cls_vocab[dim_i]))
                if isinstance(self._cls_vocab[dim_i], list):
                    self._cls_vocab[dim_i] = {str(x): i for i, x in enumerate(self._cls_vocab[dim_i])}
            self._metadata = [x for x in self._metadata if all(
                x['L'][i] in self._cls_vocab[i] or str(x['L'][i]) == '-100'
                for i in range(len(self._cls_vocab)))]
            logging.info("%d samples after filtering categories" % len(self._metadata))
        else:
            self._cls_vocab = None

        if self._world_size > 1:
            self._metadata = self._metadata[self._rank::self._world_size]
            logging.info("%d samples after sharding" % len(self._metadata))

        self._metadata.sort(key=lambda x: x['n'])
        if shuffle:
            self.rand.shuffle(self._metadata)

    def run(self):
        try:
            while True:
                self._enqueue_next_group()
        except Exception:
            logging.error(traceback.format_exc())

    def state_dict(self):
        with self._lock:
            state = {'rand': self.rand.get_state(), 'offset': self._offset, 'epoch': self._epoch}

            if hasattr(self, '_adapt_offset'):
                state['adapt_offset'] = self._adapt_offset
            logging.info("Dumped feeder state: " + str(state['offset']))
            return state

    def load_state_dict(self, state):
        logging.info("Loaded feeder state: " + str(state['offset']))
        self.rand.set_state(state['rand'])
        self._offset = state['offset']
        self._epoch = state['epoch']
        if hasattr(self, '_adapt_offset'):
            state['adapt_offset'] = self._adapt_offset

    def get_examples(self, bucket_size):
        examples = []
        with self._lock:
            for i in range(bucket_size):
                examples.append(self._get_next_example())
        return examples

    def get_batch(self):
        return self.queue.get()

    def _enqueue_next_group(self):
        tic = time.time()
        examples = self.get_examples(self._hparams.bucket_size)
        examples.sort(key=lambda x: len(x['input']))
        batches = _pack_into_batches(examples, self._batch_size, self._batch_frame_limit, self._batch_quad_frame_limit)
        if self._shuffle:
            self.rand.shuffle(batches)

        for batch in batches:
            batch = _prepare_batch(batch, tokenizer=self._tokenizer, extractor=self._extractor, hparams=self._hparams,
                                   decoder_tokenizer=self._decoder_tokenizer, cls_vocab=self._cls_vocab)
            self.queue.put(dict([(name, self.proto[name](batch[name])) for name in self.proto]))
        logging.info("Packed %d batches with %d samples in %.2f sec" % (len(batches), len(examples), time.time() - tic))

    def _get_next_example(self):
        while True:
            meta = self._metadata[self._offset]
            self._offset += 1
            if self._offset >= len(self._metadata):
                self._offset = 0
                self._epoch += 1
                if self._hparams.shuffle_training_data:
                    self.rand.shuffle(self._metadata)
            if self._filter_samples and self.skip_meta(meta):
                continue
            break

        return extract_meta(meta, self._datadir, self._hparams)

    def skip_meta(self, meta):
        if self.global_step == -1 or self.global_step >= self._hparams.data_warmup_steps:
            return False
        if self._hparams.input_length_upper_bound > 0 and \
                not self._hparams.input_length_lower_bound <= int(meta['l']) <= self._hparams.input_length_upper_bound:
            return True
        if self._hparams.target_length_lower_bound > 0 and len(meta['t']) <= self._hparams.target_length_lower_bound:
            return True
        return False

    # Methods for loading all data, not just the next batch; used for evaluation
    def _get_all_examples(self):
        examples = []
        while True:
            example = self._get_next_example()
            examples.append(example)
            if self._epoch == 1:
                self._epoch = 0
                break
        return examples

    def get_all_batches(self, exclude=None):
        examples = self._get_all_examples()
        examples = [x for x in examples if exclude is None or x['name'] not in exclude]
        examples.sort(key=lambda x: len(x['input']))

        batches = _pack_into_batches(examples, self._batch_size, self._batch_frame_limit, self._batch_quad_frame_limit,
                                     self._single)
        return batches

    def prepare_all_batches(self):
        batches = self.get_all_batches()
        ret = []
        for batch in batches:
            batch = _prepare_batch(batch, tokenizer=self._tokenizer, extractor=self._extractor, hparams=self._hparams,
                                   decoder_tokenizer=self._decoder_tokenizer, cls_vocab=self._cls_vocab)
            ret.append(batch)
        self.data = ret

    def fetch_data(self, exclude=None):
        if exclude is None:
            if hasattr(self, 'eval_batches'):
                return self.eval_batches
            data = self.data
        else:
            data = self.prepare_all_batches(self.get_all_batches(exclude))
        if self._shuffle:
            self.rand.shuffle(data)
        for batch in data:
            for name in batch:
                if name in self.proto:
                    batch[name] = self.proto[name](batch[name])
        self.eval_batches = data
        return data


def _read_meta(meta_file, format):
    meta_list = []
    for line in meta_file:
        parts = line.strip().split('|')
        if len(parts) != len(format):
            parts = line.strip().split('\t')
        if format == 'nlt':
            name, length, text = parts
            item_dict = {'n': name, 'l': int(length), 't': text}
        elif format == 'nltLa':
            name, length, text, label, _ = parts
            item_dict = {'n': name, 'l': int(length), 't': text, 'L': label.split(',')}
        else:
            raise ValueError('Invalid format for _read_meta: %s' % format)
        meta_list.append(item_dict)
    return meta_list


def _pack_into_batches(examples, batch_size, batch_frame_limit, batch_quad_frame_limit, single=False):
    batches = [[]]
    for sample in examples:
        input_len = len(sample['input'])
        if single or len(batches[-1]) == batch_size or \
                (len(batches[-1]) + 1) * input_len > batch_frame_limit or \
                (len(batches[-1]) + 1) * input_len * input_len > batch_quad_frame_limit:
            batches.append([])
        batches[-1].append(sample)
    return batches


def _prepare_batch(batch, tokenizer: Wav2Vec2CTCTokenizer, extractor: Wav2Vec2FeatureExtractor,
                   hparams, decoder_tokenizer=None, cls_vocab=None):
    inputs = extractor([x['input'] for x in batch], padding=True, pad_to_multiple_of=8, sampling_rate=hparams.sr)
    input_lengths = np.asarray([len(x['input']) for x in batch], dtype=np.int32)

    results = {'inputs': np.asarray(inputs.data['input_values']), 'input_lengths': input_lengths,
               'names': [x['name'] for x in batch]}
    if 'attention_mask' in inputs.data:
        results['input_masks'] = np.asarray(inputs.data['attention_mask'])

    if 'label' in batch[0]:
        labels = tokenizer([x['label'].upper() if hparams.upper_only else x['label'] for x in batch], padding=True)
        results['labels'] = np.asarray(labels.data['input_ids'])
        results['label_lengths'] = np.asarray([len(x['label']) for x in batch], dtype=np.int32)
        results['label_masks'] = np.asarray(labels.data['attention_mask'])

        results['labels'] = np.where(np.asarray(labels.data['attention_mask']) == tokenizer.pad_token_id, -100, results['labels'])

        if hparams.use_decoder:
            labels = decoder_tokenizer([x['label'] for x in batch], padding=True)
            results['decoder_labels'] = np.asarray(labels.data['input_ids'])
            results['decoder_label_masks'] = np.asarray(labels.data['attention_mask'])
            results['decoder_label_lengths'] = np.asarray([x.sum() for x in results['decoder_label_masks']], dtype=np.int32)

            results['decoder_labels'] = np.where(np.asarray(labels.data['attention_mask']) == decoder_tokenizer.pad_token_id, -100, results['decoder_labels'])
    if 'classifier_label' in batch[0]:
        cls_labels = []
        for x in batch:
            cls_label = [t for t in x['classifier_label']]
            for i in range(len(cls_label)):
                cls_label[i] = cls_vocab[i][cls_label[i]] if cls_label[i] != '-100' else -100
            cls_labels.append(cls_label)
        results['classifier_labels'] = np.asarray(cls_labels, dtype=np.int32)

    return results


zipfiles = {}
def get_audiofile_spec(datadir, sub_dir, name):
    if os.path.exists(os.path.join(datadir, sub_dir + '.zip')):
        if (datadir, sub_dir) not in zipfiles:
            zipfiles[(datadir, sub_dir)] = zipfile.ZipFile(os.path.join(datadir, sub_dir + '.zip'))
        file = zipfiles[(datadir, sub_dir)].open(name)
        return io.BytesIO(file.read())
    else:
        if os.path.exists(os.path.join(datadir, sub_dir, name)):
            return open(os.path.join(datadir, sub_dir, name), 'rb')
        else:
            sub_sub_dir = name[len(name.split('_')[0]) + 1:][:7]
            if os.path.exists(os.path.join(datadir, sub_dir, sub_sub_dir, name)):
                return open(os.path.join(datadir, sub_dir, sub_sub_dir, name), 'rb')
            else:
                if len(name.split('_')) == 3:
                    sub_sub_dir = name.split('_')[1]
                    return get_audiofile_spec(os.path.join(datadir, sub_dir), sub_sub_dir, name)
                elif os.path.exists(os.path.join(datadir, name)):
                    return open(os.path.join(datadir, name), 'rb')
                else:
                    raise ValueError('File not found: %s, %s, %s' % (datadir, sub_dir, name))

def get_audiofile(datadir, sub_dir, name):
    for suffix in ['.flac', '.wav', '', '.pkl']:
        try:
            audio_file = get_audiofile_spec(datadir, sub_dir, name + suffix)
            return audio_file
        except:
            pass

def read_file(f, start=0, stop=None, input_type='audio'):
    if input_type == 'audio':
        return sd.read(f, start=start, stop=stop)[0]
        # return sd.read(f, start=start, stop=stop, always_2d=True)[0].mean(-1)
    elif input_type == 'pickle':
        return pickle.load(f)
    else:
        raise ValueError('Unknown input type: %s' % input_type)


def extract_meta(meta, datadir, hparams):
    name = meta['n']
    results = {'name': name}

    audio_file = get_audiofile(datadir, name[:-len(name.split('_')[-1]) - 1], name)
    input_data = read_file(audio_file)
    results['input'] = input_data
    results['label'] = meta['t']
    results['length'] = meta['l']
    if 'L' in meta:
        results['classifier_label'] = meta['L']
    assert meta['l'] == len(input_data)

    if hparams.remove_unk:
        import re
        results['label'] = results['label'].replace('[UNK]', '')
        results['label'] = results['label'].replace('-', " ")
        results['label'] = re.sub(r'\s+', ' ', results['label']).strip()
    if hparams.replace_apos:
        results['label'] = results['label'].replace(" '", "'")

    return results


def get_input_proto(config):
    keys = {'inputs': torch.FloatTensor, 'input_lengths': torch.LongTensor,
            'labels': torch.LongTensor, 'label_lengths': torch.LongTensor,
            'names': list}
    if config.use_attention_mask:
        keys['input_masks'] = torch.LongTensor
        keys['label_masks'] = torch.LongTensor
    if config.use_decoder:
        keys['decoder_labels'] = torch.LongTensor
        keys['decoder_label_lengths'] = torch.LongTensor
        keys['decoder_label_masks'] = torch.LongTensor
    if config.use_classifier:
        keys['classifier_labels'] = torch.LongTensor
    return keys
