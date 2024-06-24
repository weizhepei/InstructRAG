DATASET=PopQA
MODEL=InstructRAG-FT # [InstructRAG-FT, InstructRAG-ICL]

CUDA_VISIBLE_DEVICES=0 python src/inference.py \
  --dataset_name $DATASET \
  --rag_model $MODEL \
  --n_docs 5 \
  --output_dir qa_results/${MODEL}/${DATASET}\
  --load_local_model \