DATASET=2WikiMultiHopQA # [PopQA, TriviaQA, NaturalQuestions, 2WikiMultiHopQA, ASQA]
MODEL=InstructRAG-ICL # [InstructRAG-FT, InstructRAG-ICL]

CUDA_VISIBLE_DEVICES=3 python src/inference.py \
  --dataset_name $DATASET \
  --rag_model $MODEL \
  --n_docs 10 \
  --output_dir qa_results/${MODEL}/${DATASET}\
  --load_local_model \
