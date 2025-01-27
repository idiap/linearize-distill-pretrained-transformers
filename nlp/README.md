# Usage

This directory contains the code to convert a pretrained HuggingFace RoBERTa model into Linformer, by replacing the 
SelfAttention layers with the LinformerSelfAttention in `linformer_utils.py` when calling 
`substitute_layer(model, config, proj_share_mode)`.
Other pretrained parameters are kept the same.
Then the model can be fine-tuned and distilled using various strategies mentioned in the paper.
The code is partially based on [LoSparse](https://github.com/yxli2123/LoSparse).

# Replicating Experiments
Experimental results in the paper can be replicated by the following commands, with arguments filled appropriately,
using the hyperparameters from the Table 4 in the paper and the table below:

| TASK_NAME    | qnli | qqp  | sst2 | imdb |
|--------------|------|------|------|--------------------------------------|
| WARMUP_STEPS | 1964 | 6823 | 1263 | 469                                  |

As for imdb, please also add `--dataset_group_name stanfordnlp` to the command.

## Std. RoBERTa

`python run_glue.py --task_name {TASK_NAME} --model_name_or_path roberta-base --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --num_train_epochs 10 --lr {LR} --num_warmup_steps {WARMUP_STEPS} --seed 0 --output_dir output/{TASK_NAME}/dense --no_prune --project_name {TASK_NAME} --with_tracking --run_name {TASK_NAME}_dense --save_checkpoint --checkpointing_steps epoch`

## Unguided

`python run_glue_linformer.py --task_name {TASK_NAME} --model_name_or_path roberta-base --teacher_path roberta-base --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --num_train_epochs 10 --lr {LR} --num_warmup_steps {WARMUP_STEPS} --alpha_output 0 --alpha_layer 0 --seed 0 --proj_share_mode {SHARE_MODE} --with_tracking --report_to wandb --run_name {TASK_NAME}_unguided`

## Target Guided
`best_step_dir` is filled with the directory of the best checkpoint from the standard RoBERTa training.

`python run_glue_linformer.py --task_name {TASK_NAME} --model_name_or_path roberta-base --teacher_path roberta-base --per_device_train_batch_size 16 --gradient_accumulation_steps 2 --teacher_ckpt output/{TASK_NAME}/dense/{best_step_dir}/pytorch_model.bin --num_train_epochs 10 --lr {LR} --num_warmup_steps {WARMUP_STEPS} --alpha_output 0 --alpha_layer {ALPHA_LD} --seed 0 --proj_share_mode {SHARE_MODE} --output_dir output/{TASK_NAME}/distill --with_tracking --report_to wandb --run_name {TASK_NAME}_distill`

## Trajectory Guided

TEACHER_LR is the LR for the teacher model from the standard RoBERTa training.

`python run_glue_linformer_joint_distill.py --task_name {TASK_NAME} --model_name_or_path roberta-base --teacher_path roberta-base --per_device_train_batch_size 16 --gradient_accumulation_steps 2 --num_train_epochs 10 --lr {LR} --teacher_lr {TEACHER_LR} --num_warmup_steps {WARMUP_STEPS} --teacher_update_cycle {T_u} --alpha_output 0 --alpha_layer {ALPHA_LD} --proj_share_mode {SHARE_MODE}  --seed 0 --output_dir output/{TASK_NAME}/traj_distill --with_tracking --report_to wandb --run_name {TASK_NAME}_traj_distill`

## Waypoint Guided

`python run_glue_linformer_snapshot_distill.py --task_name {TASK_NAME} --model_name_or_path roberta-base --teacher_path roberta-base --teacher_ckpt_pattern "output/{TASK_NAME}/dense/epoch_*" --per_device_train_batch_size 16 --gradient_accumulation_steps 2 --num_train_epochs 10 --lr {LR} --num_warmup_steps {WARMUP_STEPS} --alpha_output 0 --alpha_layer {ALPHA_LD} --proj_share_mode {SHARE_MODE} --seed 0 --output_dir output/{TASK_NAME}/waypoint_distill --with_tracking --report_to wandb --run_name {TASK_NAME}_waypoint_distill`

## Hybrid

Same as Target Guided, but with `--max_distill_steps` set to `{10000, 40000, 5000, 3000}` respectively.

