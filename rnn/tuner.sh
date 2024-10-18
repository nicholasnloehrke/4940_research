#!/bin/bash

hidden_sizes=(32 64 128 256)
layers=(1 2)
batch_sizes=(64 128 256)
learning_rates=(0.0001 0.0005 0.001)
epochs=(10 50 250)
folds=(3 5)

for hidden_size in "${hidden_sizes[@]}"; do
    for layer in "${layers[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for learning_rate in "${learning_rates[@]}"; do
                for epoch in "${epochs[@]}"; do
                    for fold in "${folds[@]}"; do
                        command="python rnn.py data/embeddings_rall_p500.pkl --hidden_size $hidden_size --layers $layer --batch_size $batch_size --learning_rate $learning_rate --epochs $epoch --folds $fold"
                        
                        echo "Running: $command"
                        output=$(eval $command)
                        echo "$output"
                    done
                done
            done
        done
    done
done

echo "Tuner done."
