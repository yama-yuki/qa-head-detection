PJ_DIR=/home/cl/yuki-yama/work/qa
DATA_DIR=${PJ_DIR}/data
OUT_DIR=${PJ_DIR}/models
SCRIPT_DIR=${PJ_DIR}/scripts

##data
data_type=squad

if data_type=squad ; then
    SQUAD_DIR=${DATA_DIR}/${data_type}
    train_data=${SQUAD_DIR}/train-v1.1.json
    dev_data=${SQUAD_DIR}/dev-v1.1.json
elif data_type=wiki ; then
    WIKI_DIR=${DATA_DIR}/${data_type}
    train_data=${WIKI_DIR}/train.json
    dev_data=${WIKI_DIR}/dev.json
fi

##model
model_name=bert-base-uncased

##hyperpatameters
epochs=2
learning_rate=3e-5
batch_size=12

##save dir
train_model_name=${model_name}_${data_type}_${epochs}_${learning_rate}_${batch_size}
output=${OUT_DIR}/${train_model_name}

mkdir ${output}

#python ${QA_DIR}/run_qa.py \    --train_file ${train_data} \--validation_file ${dev_data} \
python ${SCRIPT_DIR}/main.py \
  --model_name_or_path ${model_name} \
  --dataset_name custom \
  --do_train \
  --do_eval \
  --per_device_train_batch_size ${batch_size} \
  --learning_rate ${learning_rate} \
  --num_train_epochs ${epochs} \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ${output}