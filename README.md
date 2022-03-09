# qa-head-detection

Table of Contents
=================

<!--ts-->
   * [Description](#description)
   * [Requirements](#requirements)
   * [Usage](#usage)

<!--te-->

# Description
[1] Fine-tune BERT QA models on a custom dataset. 

```py
# main.py
# Modified to take arguments for custom datasets in main().
data_files = {'train': '', 'validation': '',}
data_files['train'] = args.custom_train_data
data_files['validation'] = args.custom_dev_data

raw_datasets = load_dataset('squad.py', data_files=data_files)
```
Write paths to the dataset in train.sh. The script only accepts .json format.
```sh
# train.sh
python ${SCRIPT_DIR}/main.py \
  --custom_train_data ${train_data} \
  --custom_dev_data ${dev_data} \
```
The train/validation files should look like this.
```
{"data": [{
  "title": "None", 
  "paragraphs": [
      {"context": "When ThouShaltNot relocated from Cleveland to Pittsburgh in 2001 , Jeremy David Long joined the band .", 
       "qas": [
          {"answers": [{"answer_start": 85, "text": "joined"}], 
           "question": "What is the head of relocated ?", "id": "170463"}]}, 
      {"context": "The process was performed the traditional way , leaving the subject in pain for a week or two thereafter .", 
       "qas": [
          {"answers": [{"answer_start": 16, "text": "performed"}], 
           "question": "What is the head of leaving ?", "id": "478389"}]}
```

[2] (Ongoing) Detect a dependency head of a word with BERT QA models.

## Requirements
- python 3.8
- pytorch 1.10
- transformers 4.16

## Usage
```sh
1. Fine-tune on a custom dataset
# Train
sh train.sh
# Predict
sh demo.sh

2. Dep-head detection
To be written
```
