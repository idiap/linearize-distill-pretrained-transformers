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

import soundfile as sd
import numpy as np
import torch
import datasets
from datasets import load_dataset

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer

np.random.seed(0)

def preprocess_sqa5(x):
    sr = 16000
    assert sr == x['document_audio']['sampling_rate']
    concat_audio = np.concatenate([x['document_audio']['array'], np.zeros([sr]), x['question_audio']['array']])
    concat_trans = x['normalized_document_text'] + ' [SEP] ' + x['normalized_question_text']
    answer_pos = (int(x['answer_spans']['start_second'][0] * sr), int(x['answer_spans']['end_second'][0] * sr))
    doc_length = len(x['document_audio']['array'])
    return {'input': concat_audio, 'label': concat_trans, 'name': x['question_id'],
            'length': len(concat_audio), 'qa_label': answer_pos, 'doc_length': doc_length}

def preprocess_voxceleb1(x):
    name = '_'.join(x['file'][:-4].split(os.path.sep)[-3:])
    return {'input': x['audio']['array'], 'classifier_label': x['clf_label'], 'label': '123',
            'name': name, 'length': len(x['audio']['array'])}

dataset_preprocessors = {('asapp/slue-phase-2', 'sqa5'): preprocess_sqa5,
                         ('s3prl/superb', 'si'): preprocess_voxceleb1}\

dataset_pre_preprocessor = {('s3prl/superb', 'si'): lambda x: x.rename_column('label', 'clf_label')}

class Feeder(threading.Thread):
    def __init__(self, dataset_id, split, processor, hparams,
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

        dataset_id = dataset_id.split('--')
        dataset = dataset_id[0]
        subset = dataset_id[1] if len(dataset_id) > 1 else None
        manual_dir = dataset_id[2] if len(dataset_id) > 2 else None
        self._dataset = load_dataset(dataset, subset, data_dir=manual_dir, split=split, trust_remote_code=True)
        if (dataset, subset) in dataset_pre_preprocessor:
            self._dataset = dataset_pre_preprocessor[(dataset, subset)](self._dataset)
        if (dataset, subset) in dataset_preprocessors:
            self._dataset = self._dataset.map(dataset_preprocessors[(dataset, subset)],
                                              num_proc=32, remove_columns=self._dataset.column_names)
        self._dataset.set_format("numpy", columns=["input"], output_all_columns=True)
        self._tokenizer = processor['tokenizer']
        self._extractor = processor['extractor']
        self._decoder_tokenizer = processor['decoder_tokenizer']

        self._batch_size = hparams.batch_size
        self._batch_frame_limit = hparams.batch_frame_limit
        self._batch_quad_frame_limit = hparams.batch_quad_frame_limit

        self._shuffle = shuffle
        self._single = single
        self._filter_samples = filter_samples

        logging.info('%d samples read' % (len(self._dataset)))
        if sample_partial != 1:
            if sample_partial < 1:
                sample_partial = int(len(self._dataset) * sample_partial)
            self._dataset = self._dataset.select(range(sample_partial))
            logging.info('Sampled %d examples' % len(self._dataset))

        hours = sum(self._dataset['length']) / hparams.sr / 3600
        logging.info('Loaded metadata for %d examples (%.2f hours)' % (len(self._dataset), hours))

        if self._world_size > 1:
            self._dataset = self._dataset.select(range(self._rank, len(self._dataset), self._world_size))
            logging.info("%d samples after sharding" % len(self._dataset))
        self._indices = list(range(len(self._dataset)))
        if shuffle:
            self.rand.shuffle(self._indices)

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
                                   decoder_tokenizer=self._decoder_tokenizer)
            self.queue.put(dict([(name, self.proto[name](batch[name])) for name in self.proto]))
        logging.info("Packed %d batches with %d samples in %.2f sec" % (len(batches), len(examples), time.time() - tic))

    def _get_next_example(self):
        while True:
            meta = self._dataset[self._indices[self._offset]]
            self._offset += 1
            if self._offset >= len(self._dataset):
                self._offset = 0
                self._epoch += 1
                if self._hparams.shuffle_training_data:
                    self.rand.shuffle(self._indices)
            if self._filter_samples and self.skip_meta(meta):
                continue
            break

        return meta

    def skip_meta(self, meta):
        if self.global_step == -1 or self.global_step >= self._hparams.data_warmup_steps:
            return False
        if self._hparams.input_length_upper_bound > 0 and \
                not self._hparams.input_length_lower_bound <= int(meta['length']) <= self._hparams.input_length_upper_bound:
            return True
        if self._hparams.target_length_lower_bound > 0 and len(meta['label']) <= self._hparams.target_length_lower_bound:
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
                                   decoder_tokenizer=self._decoder_tokenizer)
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
                   hparams, decoder_tokenizer=None):
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
    # if 'qa_label' in batch[0]:
    #     results['qa_labels'] = np.asarray([x['qa_label'] for x in batch], dtype=np.int32)
    #     results['doc_lengths'] = np.asarray([x['doc_length'] for x in batch], dtype=np.int32)
    if 'classifier_label' in batch[0]:
        results['classifier_labels'] = np.asarray([x['classifier_label'] if isinstance(x['classifier_label'], list) else [x['classifier_label']] for x in batch], dtype=np.int32)

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
