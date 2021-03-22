#!/bin/sh
export CUDA_VISIBLE_DEVICES=$1
python3 diversity_test.py
