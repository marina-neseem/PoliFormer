#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA=../data/glue_data

MODEL_TYPE=bert  # bert or roberta
MODEL_SIZE=base  # base or large
DATASET=$1  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI

MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
EPOCHS=10
if [ $MODEL_TYPE = 'bert' ]
then
  EPOCHS=3
  MODEL_NAME=${MODEL_NAME}-uncased
fi

# TARGETS="0.4"

# for TARGET in $TARGETS; do
# echo $TARGET
python -um examples.run_glue \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_NAME \
  --task_name $DATASET \
  --pretrained_model ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/raw_adapt_weight0.1/raw_adaptive0.5 \
  --do_eval \
  --do_lower_case \
  --data_dir $PATH_TO_DATA/$DATASET \
  --max_seq_length 128 \
  --per_gpu_eval_batch_size 1 \
  --per_gpu_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs $EPOCHS \
  --save_steps 0 \
  --seed 42 \
  --output_dir ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/raw_adapt_weight0.1/raw_adaptive0.5 \
  --overwrite_cache \
  --overwrite_output_dir