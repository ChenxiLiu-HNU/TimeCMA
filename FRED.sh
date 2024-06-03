#!/bin/bash

seq_lens=(36)
pred_lens=(24 36 48 60)
learning_rates=(0.0025)
channels=(32)
e_layers=(2 3)
d_layers=(2 3)
dropout_ns=(0.1)
batch_sizes=(16)

model_name="gpt2"
data_path="FRED"

for seq_len in "${seq_lens[@]}"; do 
  for pred_len in "${pred_lens[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      for channel in "${channels[@]}"; do
        for dropout_n in "${dropout_ns[@]}"; do
          for e_layer in "${e_layers[@]}"; do
            for d_layer in "${d_layers[@]}"; do
              for batch_size in "${batch_sizes[@]}"; do
                log_path="./Results/${model_name}/${data_path}/"
                mkdir -p $log_path
                log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
                nohup python train.py \
                  --data_path $data_path \
                  --device cuda:7 \
                  --batch_size $batch_size \
                  --num_nodes 107 \
                  --seq_len $seq_len \
                  --pred_len $pred_len \
                  --epochs 50 \
                  --seed 8888 \
                  --channel $channel \
                  --head 8 \
                  --learning_rate $learning_rate \
                  --dropout_n $dropout_n \
                  --e_layer $e_layer\
                  --d_layer $d_layer\
                  --model_name $model_name \
                  --num_workers 10 \
                  --d_llm 768 > $log_file &
              done
            done 
          done
        done
      done
    done
  done
done
