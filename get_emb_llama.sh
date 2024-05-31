#!/bin/bash

nohup python save_emb.py --divide train --data_path FRED --device cuda:7 --num_nodes 107 --input_len 36 --output_len 24 --model_name llama2 --d_model 4096 --l_layers 8 > ./Results/emb_time/FRED_llama_train_p5.log &
nohup python save_emb.py --divide val --data_path FRED --device cuda:7 --num_nodes 107 --input_len 36 --output_len 24 --model_name llama2 --d_model 4096 --l_layers 8 > ./Results/emb_time/FRED_llama_val_p5.log &
nohup python save_emb.py --divide test --data_path FRED --device cuda:7 --num_nodes 107 --input_len 36 --output_len 24 --model_name llama2 --d_model 4096 --l_layers 8 > ./Results/emb_time/FRED_llama_test_p5.log &
