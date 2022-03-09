# qa-head-detection

Table of Contents
=================

<!--ts-->
   * [Description](#description)
   * [Requirements](#requirements)
   * [Usage](#usage)

<!--te-->

# Description
[1] Fine-tune BERT QA models on custom datasets. 

```py
# main.py
# Modified to take arguments for custom datasets in main().
data_files = {'train': '', 'validation': '',}
data_files['train'] = args.custom_train_data
data_files['validation'] = args.custom_dev_data

raw_datasets = load_dataset('squad.py', data_files=data_files)
```
Write paths to the datasets in train.sh.
```sh
python ${SCRIPT_DIR}/main.py \
  --custom_train_data ${train_data} \
  --custom_dev_data ${dev_data} \
```

[2] (Ongoing) Detect a dependency head of a word with BERT QA models.

## Requirements
- python 3.8
- pytorch 1.10
- transformers 4.16

## Usage
```sh
1. Fine-tune on custom datasets
# Train
sh train.sh
# Predict
sh demo.sh

2. Dep-head detection
To be written
```
