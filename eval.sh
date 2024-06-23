# conda activate instructrag

export DATASET=ASQA
export CACHE_DIR=/p/llmresearch/huggingface/hub
MODEL=InstructRAG-ICL

# InstructRAG-ICL
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
  --dataset_name $DATASET \
  --model_name $MODEL \
  --n_docs 5 \
  --output_dir qa_results/${MODEL}/${DATASET}\
  --cache_dir $CACHE_DIR \
  --load_local_model \