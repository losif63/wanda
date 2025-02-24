#!/bin/bash

for LAYER in {0..31}
do
    for TYPE in "mlp" "attn" "all"
    do
        CUDA_VISIBLE_DEVICES=6 python main.py \
            --model meta-llama/Llama-2-7b-hf  \
            --prune_method wanda  \
            --sparsity_ratio 0.5  \
            --sparsity_type unstructured  \
            --save out/llama2_7b/unstructured/wanda/ \
            --eval_zero_shot \
            --modify_layer $LAYER \
            --modify_type $TYPE
    done
done