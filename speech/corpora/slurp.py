#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#

import glob
import os
import json
import tqdm
import soundfile
from matplotlib import pyplot as plt
from collections import Counter, defaultdict
import traceback as tb

# Downloaded and decompress the real and syn SLURP data to the following directories
data_dir = "database/SLURP/slurp_real/"
syn_data_dir = "database/SLURP/slurp_synth/"

# Cloned from https://github.com/pswietojanski/slurp by running
# cd database/SLURP && git clone https://github.com/pswietojanski/slurp
meta_dir= "database/SLURP/slurp/dataset/slurp/"
output_path = "data/slurp_full"

os.makedirs(output_path, exist_ok=True)
sr = 16000


def prepare_data(lang):
    os.makedirs(os.path.join(output_path, lang), exist_ok=True)
    files = {}
    for split in ['train', 'dev', 'test']:
        files[split] = open(os.path.join(output_path, "meta.%s.txt" % split), "w", encoding='utf-8')
    split_map = {'train': 'train', 'dev': 'devel', 'test': 'test'}
    audio_files = {}
    audio_dur = {}
    sec_cnt = 0
    categories = Counter()
    scenarios = Counter()
    actions = Counter()
    audio_meta = {}
    for split in ['train', 'dev', 'test']:
        meta = open(os.path.join(meta_dir, "%s.jsonl" % split_map[split])).read().splitlines()
        meta = [json.loads(line) for line in meta]
        data = []
        split_files = set()
        for m in meta:
            for file in m['recordings']:
                if file['file'] not in audio_files:
                    audio_files[file['file']] = "%s_%010d" % (lang, len(audio_files))
                    audio_meta[file['file']] = m
                else:
                    print("Found duplicate file %s" % file['file'])
                    continue
                split_files.add(file['file'])
                data.append((audio_files[file['file']], m['sentence'], m['scenario'], m['action']))
                categories[m['intent']] += 1
                scenarios[m['scenario']] += 1
                actions[m['action']] += 1
        t_sec_cnt = sec_cnt
        for file in tqdm.tqdm(split_files):
            id = audio_files[file]
            assert soundfile.info(os.path.join(data_dir, file)).samplerate == sr
            audio_dur[id] = soundfile.info(os.path.join(data_dir, file)).frames
            sec_cnt += audio_dur[id] / sr
        print("Split %s: %d files, %.2f H" % (split, len(split_files), (sec_cnt - t_sec_cnt) / 3600))

        for id, text, scenario, action in data:
            info = '|'.join([id, str(audio_dur[id]), text, scenario + ',' + action, lang])
            files[split].write(info + '\n')

    for file, id in tqdm.tqdm(audio_files.items()):
        if os.path.exists(os.path.join(output_path, lang, id + ".flac")):
            os.remove(os.path.join(output_path, lang, id + ".flac"))
        os.symlink(os.path.join(data_dir, file), os.path.join(output_path, lang, id + '.flac'))

    scenarios = sorted(list(scenarios))
    actions = sorted(list(actions))
    print("%s: Total %d files, %.2f H, %d scenarios, %d actions" %
          (lang, len(audio_files), sec_cnt / 3600, len(scenarios), len(actions)))
    plt.hist([t / sr for t in audio_dur.values()], bins=100)
    plt.title(lang)
    plt.show()
    json.dump(audio_files, open(os.path.join(output_path, lang + '.audio_files.json'), 'w'))
    scenarios = dict([(s, i) for i, s in enumerate(scenarios)])
    actions = dict([(a, i) for i, a in enumerate(actions)])
    json.dump([scenarios, actions], open(os.path.join(output_path, 'categories.json'), 'w'))


def add_synthetic_data(lang):
    fw = open(os.path.join(output_path, "meta.train.syn.txt"), "w", encoding='utf-8')
    fw.write(open(os.path.join(output_path, "meta.train.txt"), "r", encoding='utf-8').read())
    audio_files = json.load(open(os.path.join(output_path, lang + '.audio_files.json')))
    audio_dur = {}
    sec_cnt = 0
    meta = open(os.path.join(meta_dir, "train_synthetic.jsonl")).read().splitlines()
    meta = [json.loads(line) for line in meta]
    data = []
    categories = Counter()
    scenarios = Counter()
    actions = Counter()
    audio_meta = {}
    added_files = []
    for m in meta:
        for file in m['recordings']:
            if file['file'] not in audio_files:
                audio_files[file['file']] = "%s_%010d" % (lang, len(audio_files))
                audio_meta[file['file']] = m
                added_files.append(file['file'])
            else:
                print("Found duplicate file %s" % file['file'])
                continue
            data.append((audio_files[file['file']], m['sentence'], m['scenario'], m['action']))
            categories[m['intent']] += 1
            scenarios[m['scenario']] += 1
            actions[m['action']] += 1

    for file in tqdm.tqdm(added_files):
        assert soundfile.info(os.path.join(syn_data_dir, file)).samplerate == sr
        id = audio_files[file]
        audio_dur[id] = soundfile.info(os.path.join(syn_data_dir, file)).frames
        sec_cnt += audio_dur[id] / sr

    for file in tqdm.tqdm(added_files):
        id = audio_files[file]
        target_dir = os.path.join(output_path, lang, id[len(id.split('_')[0]) + 1:][:7])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        if os.path.exists(os.path.join(target_dir, id + ".flac")):
            continue
        os.symlink(os.path.join(syn_data_dir, file), os.path.join(target_dir, id + '.flac'))

    for id, text, scenario, action in data:
        info = '|'.join([id, str(audio_dur[id]), text, scenario + ',' + action, lang])
        fw.write(info + '\n')

    scenarios = sorted(list(scenarios))
    actions = sorted(list(actions))
    print("Synthetic %s: Added %d files, %.2f H, %d scenarios, %d actions" %
          (lang, len(added_files), sec_cnt / 3600, len(scenarios), len(actions)))
    plt.hist([t / sr for t in audio_dur.values()], bins=100)
    plt.title(lang)
    plt.show()
    json.dump(audio_files, open(os.path.join(output_path, lang + '.audio_files.syn.json'), 'w'))

if __name__ == '__main__':
    prepare_data('en')
    add_synthetic_data('en')