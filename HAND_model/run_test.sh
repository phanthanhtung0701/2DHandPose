#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python Hand.py \
  --input_file ./datasets/fhad_full/ \
  --test \
  --batch_size 64 \
  --model_def HandNet \
  --gpu \
  --gpu_number 0 \
  --pretrained_model ./checkpoints/fhad/model-250.pkl

