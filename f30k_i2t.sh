#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py --data_path /your_data_path --data_name f30k_bottom --vocab_path /your_vocab_path --vocab_name f30k_new_vocab.json --max_violation --bi_gru --agg_func=Mean --cross_attn=i2t --lambda_softmax=4 --num_epochs=40 --lr_update=15 --learning_rate=0.0002 --embed_size 1024 --batch_size 196