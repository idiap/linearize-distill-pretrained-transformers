#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: Apache 2.0
#

import lm_eval
import os, glob, json, shutil
import argparse
from open_instruct.lm_harness_eval import HybridMambaEvalWrapper


parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=str, default='lambada_openai,piqa,winogrande,wsc273,arc_easy,arc_challenge,sciq,logiqa')
parser.add_argument('--dry', action='store_true')
args = parser.parse_args()

task_manager = lm_eval.tasks.TaskManager()
def run_lm_eval(base_path, tasks, is_mamba=True):
    if is_mamba:
        model = HybridMambaEvalWrapper(pretrained=base_path, batch_size=1)
    else:
        model = lm_eval.models.huggingface.HFLM(pretrained=base_path)
    results = lm_eval.simple_evaluate(  # call simple_evaluate
        model=model,
        tasks=tasks,
        num_fewshot=0,
        task_manager=task_manager,

    )
    return results

subs = args.subjects.split(',')

base_path = "output/pile/"

metric_keys = {'lambada_openai': 'acc,none', 'piqa': 'acc,none', 'winogrande': 'acc,none', 'wsc273': 'acc,none',
               'arc_easy': 'acc,none', 'arc_challenge': 'acc,none', 'sciq': 'acc,none', 'logiqa': 'acc,none'}
from collections import defaultdict
results = defaultdict(dict)

models = []
models = list(glob.glob(os.path.join(base_path, "*")))
print(models)
models = [m for m in models if os.path.exists(os.path.join(m, "pytorch_model.bin"))]
print(models)
models.sort()

extra_paths = {}

for path, steps in extra_paths.items():
    for step in steps:
        models.append(os.path.join(path, f"step_{step}"))
        print(os.path.join(path, f"step_{step}"))

spec_models = ["EleutherAI/pythia-410m", "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
               "state-spaces/mamba-790m-hf", "state-spaces/mamba-1.4b-hf",]
models += spec_models


for m_i, model_path in enumerate(models):
    model_name = os.path.basename(model_path)
    if model_path in spec_models:
        os.makedirs(os.path.join(base_path, model_name), exist_ok=True)

    os.system(f"tmux rename-session '{m_i}/{len(models)} {model_name}'")
    m_subs = []
    m_results = {}
    for sub in subs:
        if model_path in spec_models:
            sub_path = os.path.join(base_path, model_name, f"{sub}.json")
        else:
            sub_path = os.path.join(model_path, f"{sub}.json")
        if os.path.exists(sub_path):
            m_results[sub] = json.load(open(sub_path))
        else:
            m_subs.append(sub)
    results[model_name] = m_results
    if args.dry:
        continue
    if not os.path.exists(os.path.join(model_path, "pytorch_model.bin")) and model_path not in spec_models:
        continue
    if m_subs:
        lm_results = run_lm_eval(model_path, m_subs, is_mamba=model_path not in spec_models)['results']
        for sub in m_subs:
            m_results[sub] = lm_results[sub]
            json.dump(m_results[sub], open(
                os.path.join(
                    model_path if model_path not in spec_models else os.path.join(base_path, model_name),
                    f"{sub}.json"), 'w'), indent=1)
    results[model_name] = m_results

print("model\t" + "\t".join(subs))
for model_path in models:
    model_name = os.path.basename(model_path)
    print(model_name, end="\t")
    lsr = []
    for key in subs:
        if model_name not in results or key not in results[model_name]:
            print(" ", end="\t")
            continue
        result = results[model_name][key]
        if isinstance(metric_keys[key], str):
            r = result[metric_keys[key]]
            if isinstance(r, list) and len(r) == 1:
                r = r[0]
            lsr.append(r)
            print(f"{r:.3f}", end="\t")
        else:
            print(f"{metric_keys[key](result):.3f}", end="\t")
            lsr.append(metric_keys[key](result))
    print("%.3f" % (sum(lsr) / len(lsr) if lsr else 0))