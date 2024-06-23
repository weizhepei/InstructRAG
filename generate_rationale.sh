# conda activate instructrag

export DATASET=PopQA
export CACHE_DIR=/p/llmresearch/huggingface

CUDA_VISIBLE_DEVICES=0 python src/inference_new.py \
  --dataset_name $DATASET \
  --model_name_or_path outputs/${DATASET} \
  --n_docs 5 \
  --output_dir qa_results/${DATASET}\
  --cache_dir $CACHE_DIR \
  --do_rationale_generation \