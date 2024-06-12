#!/bin/bash
# 获取当前主机的索引

# 模型路径作为第一个参数传入脚本
model_path=$1
host_index=$2
model_name=$3
nums_gpu=$4
echo $host_index > host_index.txt
echo $nums_gpu
# 根据任务执行相应的命令
case $host_index in
    0)
        log_file="logs/eval_arc_$model_name.log"
        echo $log_file > log_file.txt
        nohup lm_eval --model vllm \
            --model_args "pretrained=$model_path,tensor_parallel_size=$nums_gpu,dtype=float16,gpu_memory_utilization=0.9,data_parallel_size=1" \
            --tasks ai2_arc \
            --batch_size 2 > $log_file 2>&1 &
        ;;
    1)
        log_file="logs/eval_gsm8k_$model_name.log"
        echo $log_file > log_file.txt
        nohup lm_eval --model vllm \
            --model_args "pretrained=$model_path,tensor_parallel_size=$nums_gpu,dtype=float16,gpu_memory_utilization=0.9,data_parallel_size=1" \
            --tasks gsm8k \
            --batch_size auto > $log_file 2>&1 &
        ;;

    2)
        log_file="logs/eval_winogrande_$model_name.log"
        echo $log_file > log_file.txt
        nohup lm_eval --model vllm \
            --model_args "pretrained=$model_path,tensor_parallel_size=$nums_gpu,dtype=float16,gpu_memory_utilization=0.9,data_parallel_size=1" \
            --tasks winogrande \
            --batch_size 2 > $log_file 2>&1 &
        ;;
    3)
        log_file="logs/eval_truthfulqa_$model_name.log"
        echo $log_file > log_file.txt
        nohup lm_eval --model vllm \
            --model_args "pretrained=$model_path,tensor_parallel_size=$nums_gpu,dtype=float16,gpu_memory_utilization=0.9,data_parallel_size=1" \
            --tasks truthfulqa \
            --batch_size 4 > $log_file 2>&1 &
        ;;
    4)
        log_file="logs/eval_hellaswag_$model_name.log"
        echo $log_file > log_file.txt
        nohup lm_eval --model vllm \
            --model_args "pretrained=$model_path,tensor_parallel_size=$nums_gpu,dtype=float16,gpu_memory_utilization=0.9,data_parallel_size=1" \
            --tasks hellaswag \
            --batch_size 2 > $log_file 2>&1 &
        ;;
    5)
        log_file="logs/eval_mmlu_$model_name.log"
        echo $log_file > log_file.txt
        nohup lm_eval --model vllm \
            --model_args "pretrained=$model_path,tensor_parallel_size=$nums_gpu,dtype=float16,gpu_memory_utilization=0.85,data_parallel_size=1,max_model_len=512" \
            --tasks mmlu \
            --batch_size auto > $log_file 2>&1 &
        ;;
    *)
        echo "No task assigned for host $host_index"
        ;;
esac