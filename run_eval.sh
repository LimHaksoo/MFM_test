#!/bin/bash

# 학습된 LoRA 체크포인트 경로 (output_dir과 동일)
MODEL_PATH="./checkpoints/llava-v1.5-7b-mfm-lora"
# 원본 베이스 모델 (LoRA 병합을 위해 필요)
MODEL_BASE="liuhaotian/llava-v1.5-7b"
# 데이터셋 경로
DATA_ROOT="/data3/jisu/MFM/datasets/ViSA"

python eval_llava.py \
    --model_path $MODEL_PATH \
    --model_base $MODEL_BASE \
    --visa_root $DATA_ROOT \
    --batch_size 1 \
    --conv_mode llava_v1 \
    --output_dir ./eval_results_checkpoint_final > evaluation_log.txt 2>&1



# python llava/eval/mfm_eval_f1_checkpoints.py \
#   --ckpt_root ./checkpoints/llava-v1.5-7b-mfm-lora \
#   --model_base liuhaotian/llava-v1.5-7b \
#   --eval_data /data3/jisu/LLaVA/visa_llava_instruct.json \
#   --image_folder /data3/jisu/MFM/datasets/ViSA \
#   --conv_mode llava_v1 \
#   --max_new_tokens 6 \
#   --load_4bit \
#   --output_dir ./eval_results/visa_mfm