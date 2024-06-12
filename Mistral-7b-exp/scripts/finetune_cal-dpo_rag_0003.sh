
ACCELERATE_LOG_LEVEL=info


model_path=/path/zephyr-7b-sft-full
data_path=/path/ultrafeedback_binarized
output_dir=/output_models/DPO_REG_right_beta003
log_file=logs/DPO_REG_beta003.log

if [ ! -d ${output_dir} ];then
    mkdir ${output_dir}
fi

nohup deepspeed --hostfile=/tmp/mae_hostfile rag/run_dpo_rag_multi_host.py  \
    --model_name_or_path ${model_path}\
    --use_flash_attention_2 True \
    --dataset_path ${data_path} \
    --dataset_splits "train" \
    --preprocessing_num_workers 12 \
    --bf16 True \
    --loss_type "DPO_REG"\
    --ddp_timeout 5400 \
    --beta1 0.5 \
    --beta2 0.003 \
    --do_eval False \
    --evaluation_strategy "no" \
    --eval_steps 100 \
    --hub_model_id "zephyr-7b-sft-full" \
    --learning_rate 5.0e-7 \
    --log_level "info" \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --lr_scheduler_type "linear"\
    --max_length 1024 \
    --max_prompt_length 512 \
    --num_train_epochs 3 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --optim rmsprop \
    --output_dir ${output_dir} \
    --deepspeed configs/deepspeed_bak_will_bf16.json \
    --push_to_hub False \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --seed 42 \
    --warmup_ratio 0.1 > ${log_file}  2>&1 &
