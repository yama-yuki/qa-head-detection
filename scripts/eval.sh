PJ_DIR=/home/cl/yuki-yama/work/qa
MODEL_DIR=${PJ_DIR}/models
SCRIPT_DIR=${PJ_DIR}/scripts
DATA_DIR=${PJ_DIR}/data

## mode
mode=test

## finetuned model
model_dir=${MODEL_DIR}/bert-base-uncased_tok_2_3e-5_24

## model
pretrained_model=bert-base-uncased

## test data path (.json)
test_path=${DATA_DIR}/wiki/advcl_tok_test.json

CUDA_VISIBLE_DEVICES=0 python ${SCRIPT_DIR}/predict.py \
  --mode ${mode} \
  --model_dir ${model_dir} \
  --pretrained ${pretrained_model} \
  --test_path ${test_path}