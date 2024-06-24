DATASET=PopQA # [PopQA, TriviaQA, NaturalQuestions, 2WikiMultiHopQA, ASQA]
MODEL=InstructRAG-FT # [InstructRAG-FT, InstructRAG-ICL]

CUDA_VISIBLE_DEVICES=0 python src/inference.py \
  --dataset_name $DATASET \
  --rag_model $MODEL \
  --n_docs 5 \
  --output_dir eval_results/${MODEL}/${DATASET}
  # --load_local_model # Uncomment this line if you want to load a local model