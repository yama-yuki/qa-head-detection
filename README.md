# qa-head-detection

Table of Contents
=================

<!--ts-->
   * [Description](#description)
   * [Requirements](#requirements)
   * [Usage](#usage)

<!--te-->

# Description
[1] Fine-tune BERT QA models on custom datasets. (need to change train.sh)

```py
#Modified to take arguments for custom datasets in main.py main().
data_files = {'train': '', 'validation': '',}
data_files['train'] = args.custom_train_data
data_files['validation'] = args.custom_dev_data

raw_datasets = load_dataset('squad.py', data_files=data_files)
```

[2] (Ongoing) Detect a dependency head of a word with BERT QA models.

## Requirements
- python 3.8
- pytorch 1.10
- transformers 4.16

## Usage
```sh
1. Fine-tune on a custom dataset
# Clone repository
git clone https://github.com/yama-yuki/qa-head-detection.git
# Create conda environment
conda create -n qa python=3.8
# Activate conda environment
conda activate qa
# Fine-tune BERT
sh train.sh
# Predict
sh demo.sh

2. Dep-head detection
To be written
```
