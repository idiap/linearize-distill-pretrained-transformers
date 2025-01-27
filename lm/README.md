# Usage

This directory contains the code to convert a pretrained HuggingFace GPTNeoX model (e.g. Pythia) into a Mamba-based model.
A GPTNeoXMambaModel class is implemented in `modeling_hybrid_mamba.py`, that replicates the GPTNeoXModel in HF but 
replaces the attention layers with Mamba mixers.
Then a pretrained HF LM can be converted by calling `model = build_gpt_neox_mamba_model_from_gpt_neox(model)`, that 
builds a GPTNeoXMambaModel for causal LM, and loads the pretrained parameters into it.
Then the model can be fine-tuned and distilled using various strategies mentioned in the paper.
The code is partially based on [open-instruct](https://github.com/allenai/open-instruct).

# Replicating Experiments

We use the pre-shuffled Pile corpus available at [HF datasets](https://huggingface.co/datasets/EleutherAI/pile-standard-pythia-preshuffled).
Since we only use a small portion of the data for conversion, we only download the `document.idx
` and `document-00000-of-00020.bin` files to `data/pile_pythia/document` for the experiments.

Then the experiments can be replicated by the following commands, with arguments filled according to the table below.

| DATA_PROP     | 05p    | 1p      | 2p      |
|---------------|--------|---------|---------|
| TRAIN_SAMPLES | 732160 | 1464320 | 2928640 |
| 40P           | 4576   | 9120    | 18280   |
| DECAY_STEPS   | 12000  | 24000   | 24000   |

The `lm` directory needs to be added to PYTHONPATH first. As for the unguided mode, run:

``
python open_instruct/retrain_mamba_distill.py --model_name_or_path EleutherAI/pythia-1b --tokenizer_name EleutherAI/pythia-1.4b --train_file data/pile_pythia/document --preprocessing_num_workers 64 --per_device_train_batch_size 8 --gradient_accumulation_steps 16 --learning_rate 3e-4 --lr_scheduler_type cosine --warmup_ratio 0.03 --num_train_epochs 2 --report_to wandb --logging_steps 10 --run_name LM_direct_{DATA_PROP} --use_tf32 --mixed_precision fp16 --use_flash_attn --use_mamba --alpha_layer_distill_loss 0 --clip_grad_norm 1 --output_dir output/pile/ --weight_decay_mamba_only --weight_decay 0.1 --checkpointing_steps 0.01 --max_train_samples {TRAIN_SAMPLES} --alpha_ce_loss 1 --alpha_logits_distill_loss 0 --with_tracking --min_lr 3e-5 --lr_decay_steps {DECAY_STEPS}
``

For the target guided mode, run:

``
python open_instruct/retrain_mamba_distill.py --model_name_or_path EleutherAI/pythia-1b --teacher_name_or_path EleutherAI/pythia-1b --tokenizer_name EleutherAI/pythia-1.4b --train_file data/pile_pythia/document --preprocessing_num_workers 64 --per_device_train_batch_size 8 --gradient_accumulation_steps 16 --learning_rate 3e-4 --lr_scheduler_type cosine --warmup_ratio 0.03 --num_train_epochs 2 --report_to wandb --logging_steps 10 --run_name LM_{DATA_PROP} --use_tf32 --mixed_precision fp16 --use_flash_attn --use_mamba --alpha_layer_distill_loss 15 --clip_grad_norm 1 --output_dir output/pile/ --weight_decay_mamba_only --weight_decay 0.1 --checkpointing_steps 0.01 --max_train_samples {TRAIN_SAMPLES} --alpha_ce_loss 1 --alpha_logits_distill_loss 0 --with_tracking --min_lr 3e-5 --lr_decay_steps {DECAY_STEPS}
``

Then for the hybrid modes we simply continue the training from 40% steps with alpha set to 0 instead, by copying the 
checkpoints from `output/pile/LM_{DATA_PROP}/steps_{40P}` to `output/pile/LM_{DATA_PROP}_hybrid` and running:

``
python open_instruct/retrain_mamba_distill.py --model_name_or_path EleutherAI/pythia-1b --tokenizer_name EleutherAI/pythia-1.4b --train_file data/pile_pythia/document --preprocessing_num_workers 64 --per_device_train_batch_size 8 --gradient_accumulation_steps 16 --learning_rate 3e-4 --lr_scheduler_type cosine --warmup_ratio 0.03 --num_train_epochs 2 --report_to wandb --logging_steps 10 --run_name LM_{DATA_PROP}_hybrid --use_tf32 --mixed_precision fp16 --use_flash_attn --use_mamba --alpha_layer_distill_loss 0 --clip_grad_norm 1 --output_dir output/pile/ --weight_decay_mamba_only --weight_decay 0.1 --checkpointing_steps 0.01 --max_train_samples {TRAIN_SAMPLES} --alpha_ce_loss 1 --alpha_logits_distill_loss 0 --with_tracking --min_lr 3e-5 --lr_decay_steps {DECAY_STEPS} --resume_from_checkpoint True
``

Then the models can be evaluated by running `eval/run_lm_evals.py`.