# qa-head-detection

Table of Contents
=================

<!--ts-->
   * [Description](#description)
   * [Requirements](#requirements)
   * [Usage](#usage)

<!--te-->

# Description
[1] For fine-tuning BERT QA models on custom datasets. (need to modify train.sh)

[2] (Ongoing) For detecting a dependency head of a word with BERT QA models.

## Requirements
- python 3.8
- pytorch 1.10.2
- transformers 4.16.2

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
