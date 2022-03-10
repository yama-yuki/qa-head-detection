PJ_DIR=/home/cl/yuki-yama/work/qa
MODEL_DIR=${PJ_DIR}/models
SCRIPT_DIR=${PJ_DIR}/scripts
DATA_DIR=${PJ_DIR}/data

## mode
mode=test
## snt/tok
type=tok

## model
pretrained_model=bert-base-uncased
## finetuned model
model_dir=${MODEL_DIR}/bert-base-uncased_${type}_2_3e-5_12
## test data path (.json)
test_path=${DATA_DIR}/wiki/advcl_${type}_test.json

CUDA_VISIBLE_DEVICES=0 python ${SCRIPT_DIR}/predict.py \
  --mode ${mode} \
  --model_dir ${model_dir} \
  --pretrained ${pretrained_model} \
  --test_path ${test_path}