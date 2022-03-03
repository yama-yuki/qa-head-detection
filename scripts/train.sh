PJ_DIR=/home/cl/yuki-yama/work/qa
DATA_DIR=${PJ_DIR}/data/wiki
OUT_DIR=${PJ_DIR}/models
SCRIPT_DIR=${PJ_DIR}/scripts

##data squad/snt/tok (for naming the model)
data_type=snt

##data path
train_data=${DATA_DIR}/advcl_train.json
dev_data=${DATA_DIR}/advcl_dev.json

##model
model_name=bert-base-uncased

##hyperpatameters
epochs=2
learning_rate=3e-5
batch_size=24

##save dir
train_model_name=${model_name}_${data_type}_${epochs}_${learning_rate}_${batch_size}
#output=${OUT_DIR}/${train_model_name}
output=${OUT_DIR}/test

mkdir -p ${output}

#CUDA_VISIBLE_DEVICES=0 
python ${SCRIPT_DIR}/main.py \
  --model_name_or_path ${model_name} \
  --dataset_name ${data_type} \
  --custom_train_data ${train_data} \
  --custom_dev_data ${dev_data} \
  --per_device_train_batch_size ${batch_size} \
  --learning_rate ${learning_rate} \
  --num_train_epochs ${epochs} \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ${output}