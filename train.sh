DATASET=PopQA
PER_DEVICE_BATCH_SIZE=1
NUM_DEVICE=4
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_DEVICE/$PER_DEVICE_BATCH_SIZE))

CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=$NUM_DEVICE src/finetune.py \
  --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset_name $DATASET \
  --output_dir saved_checkpoints/InstructRAG-FT/${DATASET} \
  --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
  --num_train_epochs 2 \
  --n_docs 5 \
  --learning_rate 2.5e-5 \
  --lr_scheduler_type "cosine" \
  --bf16 True \
  --tf32 True \
  --logging_steps 1 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --seed 42 \
  --model_max_length 4096 \
  --ddp_timeout 1800 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer"