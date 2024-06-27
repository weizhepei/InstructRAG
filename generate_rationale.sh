DATASET=PopQA

CUDA_VISIBLE_DEVICES=0 python src/inference.py \
  --dataset_name $DATASET \
  --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
  --n_docs 5 \
  --output_dir dataset/${DATASET}\
  --do_rationale_generation \