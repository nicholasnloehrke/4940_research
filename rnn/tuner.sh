#!/bin/bash

hidden_sizes=(32 64 128)
layers=(1 2)
batch_sizes=(64 128 256)
learning_rates=(0.0001 0.0005 0.001)
epochs=(2 4)
folds=(3 5)

# Loop through all combinations of parameters
for hidden_size in "${hidden_sizes[@]}"; do
    for layer in "${layers[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for learning_rate in "${learning_rates[@]}"; do
                for epoch in "${epochs[@]}"; do
                    for fold in "${folds[@]}"; do
                        # Construct the command
                        command="python rnn.py data/embeddings_rall_p500.pkl --hidden_size $hidden_size --layers $layer --batch_size $batch_size --learning_rate $learning_rate --epochs $epoch --folds $fold"
                        
                        # Execute the command and capture the output
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
