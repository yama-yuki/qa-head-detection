PJ_DIR=/home/cl/yuki-yama/work/qa
MODEL_DIR=${PJ_DIR}/models
SCRIPT_DIR=${PJ_DIR}/scripts

##finetuned_model
model_dir=${MODEL_DIR}/bert-base-uncased_tok_2_3e-5_24

##model
pretrained_model=bert-base-uncased

##tsv data path
input_path=${SCRIPT_DIR}/demo/input.tsv
output_path=${SCRIPT_DIR}/demo/output.tsv

CUDA_VISIBLE_DEVICES=0 python ${SCRIPT_DIR}/predict.py \
  --mode pred \
  --model_dir ${model_dir} \
  --pretrained ${pretrained_model} \
  --input_path ${input_path} \
  --output_path ${output_path}