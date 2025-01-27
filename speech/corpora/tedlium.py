#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#

import json
import os, glob
import librosa
import soundfile as sd
import tqdm
import traceback as tb
import re
import inflect
import unidecode
sr = 16000

# Please download the TEDLIUMv3 dataset (https://www.openslr.org/51/) to the following path
tedlium_db_path = "database/TEDLIUM_release-3/legacy"

output_path = "data/tedlium"


_inflect = inflect.engine()

def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            s = 'two thousand'
        elif num > 2000 and num < 2010:
            s = 'two thousand ' + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            s = _inflect.number_to_words(num // 100) + ' hundred'
        else:
            s = _inflect.number_to_words(
                num, andword='', zero='oh', group=2).replace(', ', ' ')
    else:
        s = _inflect.number_to_words(num, andword='')
    return ' ' + s + ' '


def _expand_decimal_point(m):
    return m.group(1).replace('.', ' point ')

def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))

def _expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'

def normalize(text):
    _decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
    _dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
    _ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
    _number_re = re.compile(r'[0-9]+')
    _whitespace_re = re.compile(r'\s+')

    text = text.replace('[', '').replace(']', '').\
        replace('&', 'and').replace('%', 'percent').replace('+', 'plus').replace('@', 'at').replace("<unk>", "[UNK]")
    text = unidecode.unidecode(text)

    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)

    text = re.sub(_whitespace_re, ' ', text)
    return text


# An example of each line in .stm file:
# AlGore_2009 1 Al_Gore 23.46 40.05 <F0_M> but this understates the seriousness of this particular problem because it doesn 't show the thickness of the ice the arctic ice cap is in a sense the beating heart of the global climate system it expands in winter and contracts in summer the next slide i show you will be
def prepare_segment():
    os.makedirs(output_path, exist_ok=True)
    vocab = set("abcdefghijklmnopqrstuvwxyz")
    for split in ['train', 'dev', 'test']:
        fw = open(os.path.join(output_path, 'meta.%s.txt' % split), 'w')
        files = list(glob.iglob(os.path.join(tedlium_db_path, split, 'stm', '*.stm')))
        for f in tqdm.tqdm(files):
            filename = os.path.split(f)[-1][:-4]
            os.makedirs(os.path.join(output_path, filename), exist_ok=True)
            audio, _ = librosa.load(os.path.join(tedlium_db_path, split, 'sph', filename + '.sph'), sr=sr)
            lines = open(f).read().splitlines()

            for i, l in enumerate(lines):
                l = l.split(' ')
                assert filename == l[0] and l[1] == '1'
                starts = float(l[3])
                ends = float(l[4])
                l = ' '.join(l[6:])
                if l.lower() == 'ignore_time_segment_in_scoring':
                    continue
                l = normalize(l)
                # for ch in l:
                #     if ch not in vocab:
                #         # vocab.add(ch)
                #         print(ch)
                #         # print(ol, l)
                #         break
                name = '%s_%08d' % (filename, i)

                starts = int(starts * sr)
                ends = int(ends * sr)

                fw.write('|'.join([name, str(ends - starts), l]) + '\n')
                try:
                    sd.write(os.path.join(output_path, filename, name + '.flac'), audio[starts: ends], sr)
                except:
                    print("Failed:", name)
                    tb.print_exc()
                    continue
    vocab = ['[PAD]'] + sorted(list(vocab)) + ["'", '|', '[UNK]']
    vocab = dict([(c, i) for i, c in enumerate(vocab)])
    open(os.path.join(output_path, 'vocab.json'), 'w').write(json.dumps(vocab, indent=1))

if __name__ == '__main__':
    prepare_segment()
