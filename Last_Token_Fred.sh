#!/bin/bash

nohup python store_emb.py --divide train --data_path FRED --device cuda:7 --num_nodes 107 --input_len 36 --output_len 24 > ./Results/emb_time/FRED_train_p5.log &
nohup python store_emb.py --divide val --data_path FRED --device cuda:7 --num_nodes 107 --input_len 36 --output_len 24 > ./Results/emb_time/FRED_val_p5.log &
nohup python store_emb.py --divide test --data_path FRED --device cuda:7 --num_nodes 107 --input_len 36 --output_len 24 > ./Results/emb_time/FRED_test_p5.log &
