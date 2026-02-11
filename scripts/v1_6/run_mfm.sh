#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

DEEPSPEED_CONFIG="./scripts/zero3.json"

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path liuhaotian/llava-v1.6-vicuna-7b \
    --version v1 \
    --data_path /data3/jisu/LLaVA/visa_llava_instruct.json \
    --image_folder /data3/jisu/MFM/datasets/ViSA \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --fp16 True \
    --output_dir ./checkpoints/llava-v1.6-vicuna-7b-mfm-lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4\
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4\
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none > training_log_v1_6.txt 2>&1

# # [변경 2] v1.6의 핵심 기능인 'AnyRes' (동적 해상도) 활성화
# --image_aspect_ratio anyres \
# # [변경 3] AnyRes 사용 시 추가 권장되는 파라미터 (공간적 unpad)
# --mm_patch_merge_type spatial_unpad \