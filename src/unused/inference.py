from vllm import LLM, SamplingParams
import os, sys
import argparse
import json
# from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
import data_utils
import common_utils

parser = argparse.ArgumentParser(description='Decode VLLM')
parser.add_argument('--data_dir', type=str, default="/p/llmresearch/tqf5qb/projects/rag_tune/dataset/rag_nq",
                    help='Directory containing the data')
parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                    help='Path to the LLM model or name of the model in Hugging Face model hub')
parser.add_argument('--download_dir', type=str, default="/p/llmresearch/huggingface/hub",
                    help='Directory to cached models')
parser.add_argument('--temperature', type=float, default=0,
                    help='Temperature for sampling')
parser.add_argument('--max_tokens', type=int, default=4096,
                    help='Maximum number of tokens to generate')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
parser.add_argument('--max_instances', type=int, default=sys.maxsize)
parser.add_argument('--prompt_dict_path', type=str, default="/p/llmresearch/tqf5qb/projects/rag_tune/examples/prompts/rag.json")

args = parser.parse_args()
print(args)


def rationale_generation(dataset_name, dataset2retriever, data_split, max_instances=-1, num_device=-1, model_size='8b'):
    if model_size == '8b':
        model = 'llama_3_8b_instruct'
        args.model = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_size == '70b':
        model = 'llama_3_70b_instruct'
        args.model = "meta-llama/Meta-Llama-3-70B-Instruct"
    else:
        raise ValueError('Invalid model size')

    if dataset_name != 'NaturalQuestions':
        args.prompt_dict_path = '/p/llmresearch/tqf5qb/projects/rag_tune/examples/prompts/rag_tqa.json'

    args.max_tokens=2048
    if max_instances > 0:
        args.max_instances = max_instances
    num_docs_test=5 if dataset_name != '2WikiMultiHopQA' else 10
    retriever_name = dataset2retriever[dataset_name]
    args.data_dir = f'/p/llmresearch/tqf5qb/datasets/{dataset_name}/{retriever_name}_retrieval'

    # Zero-shot for long answer generation
    data_path = os.path.join(args.data_dir, data_split + '.json')
    output_file = args.data_dir + f'/with_rationale/generator_{model_size}/{data_split}.json'
            
    print(f"Loading eval set from: {data_path}")
    with open(data_path, "r") as fin:
        data = json.load(fin)

    data = data[:args.max_instances]

    if num_device == -1:
        num_device = 2 if '70B' in args.model else 1      
    llm = LLM(model=args.model, tensor_parallel_size=num_device)
    tokenizer = llm.get_tokenizer()

    prompts, list_dict_data, num_docs_list, metadata = data_utils.format_prompt_with_data_frame(
        df=data,
        dataset_name=dataset_name,
        prompt_dict=common_utils.jload(args.prompt_dict_path),
        tokenizer=tokenizer,
        random_n_docs=False,
        n_docs=num_docs_test,
        do_ans_extraction=False,
        do_sample_selection=False,
        zero_shot_long_ans_generation=True,
        no_retrieval=False,
        use_zero_shot_long=False,
    )

    sampling_params = SamplingParams(temperature=args.temperature, 
                                    top_p=args.top_p, 
                                    max_tokens=args.max_tokens, 
                                    seed=args.seed,
                                    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    outputs = llm.generate(prompts, sampling_params)


    # Save the outputs as a JSON file.
    output_data = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        dict_data = list_dict_data[i]
        do_ans_extraction=False
        do_sample_selection=False
        num_doc=num_docs_test
        output_data.append({
            # "instruction": dict_data["instruction"] if "instrcution" in dict_data else None,
            "question": dict_data["question"],
            "true_answers": dict_data["answers"] if "answers" in dict_data else dict_data["true_answers"],
            "qa_pairs": dict_data["qa_pairs"] if "qa_pairs" in dict_data else None,
            "topic": dict_data["topic"] if "topic" in dict_data else None,
            "output": generated_text,
            "logits_output": None,
            "pred_long_ans": generated_text,
            "prompt": prompt,
            "decoder_name_or_path": args.model,
            "sample_mode": sampling_params,
            "ctxs": dict_data["ctxs"][:num_doc][::-1] if (dict_data["ctxs"][0]['score'] > dict_data["ctxs"][1]['score']) else dict_data["ctxs"][:num_doc],
            })

    utils.jdump(output_data, output_file)
    print(f"Outputs saved to {output_file}")


def instruct_rag_ft(dataset_name, num_docs_test, dataset2retriever, learning_rate, use_long_demonstration=False, target_task=None):
    # args.max_instances=20
    args.max_tokens = 4096
    retriever_name = dataset2retriever[dataset_name]

    # model_path=f'../outputs/{dataset_name}/test'
    model_path = 'meng-lab/NaturalQuestions-InstructRAG-FT'

    data_path = f'/p/llmresearch/tqf5qb/datasets/{dataset_name}/{retriever_name}_retrieval/test.json'

    if dataset_name != 'NaturalQuestions':
        args.prompt_dict_path = '/p/llmresearch/tqf5qb/projects/rag_tune/examples/prompts/rag_tqa.json'


    output_file = f'../qa_results/{dataset_name}/result.json'


    print(f"Loading eval set from: {data_path}")
    with open(data_path, "r") as fin:
        data = json.load(fin)
        
    data = data[:args.max_instances]

    llm = LLM(model=model_path)
    tokenizer = llm.get_tokenizer()

    prompts, list_dict_data, num_docs_list, metadata = data_utils.format_prompt_with_data_frame(
        df=data,
        dataset_name=dataset_name,
        prompt_dict=common_utils.jload(args.prompt_dict_path),
        tokenizer=tokenizer,
        random_n_docs=False,
        n_docs=num_docs_test,
        do_ans_extraction=False,
        do_sample_selection=False,
        use_long_demonstration=use_long_demonstration,
        demos=[],
        zero_shot_long_ans_generation=False,
        no_retrieval=False,
        use_zero_shot_long=False,
    )

    sampling_params = SamplingParams(temperature=args.temperature, 
                                    max_tokens=args.max_tokens, 
                                    seed=args.seed,
                                    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    outputs = llm.generate(prompts, sampling_params)

    # Save the outputs as a JSON file.
    output_data = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        dict_data = list_dict_data[i]
        output_data.append({
            "question": dict_data["question"],
            "true_answers": dict_data["answers"] if "answers" in dict_data else dict_data["true_answers"],
            "qa_pairs": dict_data["qa_pairs"] if "qa_pairs" in dict_data else None,
            "topic": dict_data["topic"] if "topic" in dict_data else None,
            "output": generated_text,
            "logits_output": None,
            "zero_shot_long_ans": dict_data['pred_long_ans'] if 'pred_long_ans' in dict_data else None,
            "prompt": prompt,
            "decoder_name_or_path": args.model,
            "sample_mode": sampling_params,
            "ctxs": dict_data["ctxs"][:num_docs_test][::-1] if (dict_data["ctxs"][0]['score'] > dict_data["ctxs"][1]['score']) else dict_data["ctxs"][:num_docs_test],
            })

    common_utils.jdump(output_data, output_file)
    print(f"Outputs saved to {output_file}")



def in_context_ralm(model_size, num_docs_test, dataset_name, dataset2retriever, use_zero_shot_cot, use_long_demonstration, use_short_demonstration, data_split='test', max_instances=-1, num_device=-1, num_demo=-1, demo_seed=None, target_task=None, demo_id=None):
    if max_instances > 0:
        args.max_instances = max_instances
    if model_size == '8b':
        model = 'llama_3_8b_instruct'
        args.model = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_size == '70b':
        model = 'llama_3_70b_instruct'
        args.model = "meta-llama/Meta-Llama-3-70B-Instruct"
    else:
        raise ValueError('Invalid model size')

    if target_task is None:
        if dataset_name != 'NaturalQuestions':
            args.prompt_dict_path = '/p/llmresearch/tqf5qb/projects/rag_tune/examples/prompts/rag_tqa.json'
    else:
        if target_task != 'NaturalQuestions':
            args.prompt_dict_path = '/p/llmresearch/tqf5qb/projects/rag_tune/examples/prompts/rag_tqa.json'

    args.max_tokens=2048
    retriever_name = dataset2retriever[dataset_name]
    if target_task is not None:
        data_path = f'/p/llmresearch/tqf5qb/datasets/{target_task}/{dataset2retriever[target_task]}_retrieval/test.json'
    else:
        args.data_dir=f'/p/llmresearch/tqf5qb/datasets/{dataset_name}/{retriever_name}_retrieval'
        data_path = os.path.join(args.data_dir, data_split + '.json')

    if demo_seed is None and num_demo == -1:
        # with open(f'/p/llmresearch/tqf5qb/datasets/InstructRAG/{dataset_name}/{retriever_name}_retrieval/with_rationale/ablation_study/without_gt/selected_demos.json', 'r') as fin:
        # with open(f'/p/llmresearch/tqf5qb/datasets/InstructRAG/{dataset_name}/{retriever_name}_retrieval/with_rationale/ablation_study/without_gt/selected_demos_{demo_id}.json', 'r') as fin:
        # with open(f'/p/llmresearch/tqf5qb/datasets/InstructRAG/{dataset_name}/{retriever_name}_retrieval/with_rationale/70b_generator/selected_demos.json', 'r') as fin:
        with open(f'/p/llmresearch/tqf5qb/datasets/InstructRAG/{dataset_name}/{retriever_name}_retrieval/with_rationale/template_rationale/selected_demos.json', 'r') as fin:
            demo_data = json.load(fin)
            demos = [{'question': item['question'], 'true_answers': item['true_answers'], 'output': item['output']} for item in demo_data]
    elif demo_seed is None and num_demo >= 0:
        if num_demo != 0:
            with open(f'/p/llmresearch/tqf5qb/datasets/InstructRAG/{dataset_name}/{retriever_name}_retrieval/with_rationale/demo_sensitivity/set_1/selected_demos_num_{num_demo}.json', 'r') as fin:
                    demo_data = json.load(fin)
                    demos = [{'question': item['question'], 'true_answers': item['true_answers'], 'output': item['output']} for item in demo_data]
        else:
            demos = []
    else:
        with open(f'/p/llmresearch/tqf5qb/datasets/InstructRAG/{dataset_name}/{retriever_name}_retrieval/with_rationale/selected_demos_seed_{demo_seed}.json', 'r') as fin:
            demo_data = json.load(fin)
            demos = [{'question': item['question'], 'true_answers': item['true_answers'], 'output': item['output']} for item in demo_data]

    if num_docs_test == 0:
        mode = 'vanilla_prompting'
        assert sum([use_zero_shot_cot, use_long_demonstration, use_short_demonstration]) == 0
    elif use_zero_shot_cot:
        mode = 'zero_shot_cot'
        assert sum([use_zero_shot_cot, use_long_demonstration, use_short_demonstration]) == 1
    elif use_long_demonstration:
        mode = 'instruct_rag_demonstration'
        assert sum([use_zero_shot_cot, use_long_demonstration, use_short_demonstration]) == 1
    elif use_short_demonstration:
        mode = 'vanilla_demonstration'
        assert sum([use_zero_shot_cot, use_long_demonstration, use_short_demonstration]) == 1
    else:
        mode = 'in_context_ralm'
        assert sum([use_zero_shot_cot, use_long_demonstration, use_short_demonstration]) == 0
    
    if demo_seed is None and num_demo == -1:
        if target_task is not None:
            output_file = f'/p/llmresearch/tqf5qb/projects/rag_tune/new_qa_results/{dataset_name}_transfer2_{target_task}/{dataset2retriever[target_task]}_retrieval/{model}/{mode}/num_doc_{num_docs_test}/result.json'
        else:
            output_file = f'/p/llmresearch/tqf5qb/projects/rag_tune/new_qa_results/{dataset_name}/{retriever_name}_retrieval/{model}/{mode}/ablation_study/without_gt/num_doc_{num_docs_test}/result.json'
            # output_file = f'/p/llmresearch/tqf5qb/projects/rag_tune/new_qa_results/{dataset_name}/{retriever_name}_retrieval/{model}/{mode}/ablation_study/without_gt/num_doc_{num_docs_test}/demo_{demo_id}/result.json'
            # output_file = f'/p/llmresearch/tqf5qb/projects/rag_tune/new_qa_results/{dataset_name}/{retriever_name}_retrieval/{model}/{mode}/70b_generator/num_doc_{num_docs_test}/result.json'
    elif demo_seed is None and num_demo >= 0:
        output_file = f'/p/llmresearch/tqf5qb/projects/rag_tune/new_qa_results/{dataset_name}/{retriever_name}_retrieval/{model}/{mode}/demo_sensitivity/set_1/demo_num_{num_demo}/result.json'
    else:
        output_file = f'/p/llmresearch/tqf5qb/projects/rag_tune/new_qa_results/{dataset_name}/{retriever_name}_retrieval/{model}/{mode}/seed_{demo_seed}/result.json'
            
    print(f"Loading eval set from: {data_path}")
    with open(data_path, "r") as fin:
        data = json.load(fin)

    print(f"Outputs will be saved to {output_file}")

    data = data[:args.max_instances]

    if num_device == -1:
        num_device = 2 if '70B' in args.model else 1
    print(f'Loading model {args.model}...')         
    llm = LLM(model=args.model, tensor_parallel_size=num_device)
    print('Model loaded!')
    tokenizer = llm.get_tokenizer()

    prompts, list_dict_data, num_docs_list, metadata = data_preprocessor.format_prompt_with_data_frame(
        df=data,
        dataset_name=dataset_name,
        prompt_dict=utils.jload(args.prompt_dict_path),
        tokenizer=tokenizer,
        random_n_docs=False,
        n_docs=num_docs_test,
        do_ans_extraction=False,
        do_sample_selection=False,
        use_zero_shot_cot=use_zero_shot_cot,
        demos=demos,
        use_long_demonstration = use_long_demonstration,
        use_short_demonstration = use_short_demonstration,
        zero_shot_long_ans_generation=False,
        no_retrieval= num_docs_test == 0,
        use_zero_shot_long=False,
    )

    sampling_params = SamplingParams(temperature=args.temperature, 
                                    max_tokens=args.max_tokens, 
                                    seed=args.seed,
                                    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    outputs = llm.generate(prompts, sampling_params)


    # Save the outputs as a JSON file.
    output_data = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        dict_data = list_dict_data[i]
        do_ans_extraction=False
        do_sample_selection=False
        num_doc=num_docs_test
        output_data.append({
            # "instruction": dict_data["instruction"] if "instrcution" in dict_data else None,
            "question": dict_data["question"],
            "true_answers": dict_data["answers"] if "answers" in dict_data else dict_data["true_answers"],
            "qa_pairs": dict_data["qa_pairs"] if "qa_pairs" in dict_data else None,
            "topic": dict_data["topic"] if "topic" in dict_data else None,
            "output": generated_text,
            "logits_output": None,
            "pred_long_ans": generated_text,
            "prompt": prompt,
            "decoder_name_or_path": args.model,
            "sample_mode": sampling_params,
            "ctxs": dict_data["ctxs"][:num_doc][::-1] if (len(dict_data["ctxs"]) > 0 and dict_data["ctxs"][0]['score'] > dict_data["ctxs"][1]['score']) else dict_data["ctxs"][:num_doc],
            })

    utils.jdump(output_data, output_file)

    if dataset_name == 'Bio':
        # check dir exists
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        output_file = output_file.replace('.json', '.jsonl')
        with open(output_file, 'w') as f:
            for item in output_data:
                d = {'topic': item['topic'], 'output': item['output']}
                f.write("%s\n" % json.dumps(d))
        print(f"Outputs (.jsonl) saved to {output_file}")

# CUDA_VISIBLE_DEVICES='0' python vllm_decode.py & 
if __name__ == "__main__":

    datasets = ['PopQA', 'TriviaQA-unfiltered', 'NaturalQuestions', 'Bio', '2WikiMultiHopQA', 'ASQA']

    dataset2retriever = {
        'PopQA': 'contriever',
        'TriviaQA-unfiltered': 'contriever',
        'NaturalQuestions': 'dpr',
        '2WikiMultiHopQA': 'bm25',
        'Bio': 'contriever',
        'ASQA': 'gtr',
        }

    dataset2topk = {
        'PopQA': 5,
        'TriviaQA-unfiltered': 5, # best_demo_seed = 87
        'NaturalQuestions': 5,
        '2WikiMultiHopQA': 10,
        'Bio': 5,
        'ASQA': 5}
    
    # Rationale Generation
    # rationale_generation(dataset_name='NaturalQuestions', dataset2retriever=dataset2retriever, data_split='train', model_size='70b', max_instances=-1)


    # Instruct-RAG-FT Model
    instruct_rag_ft(dataset_name='PopQA', num_docs_test=5, dataset2retriever=dataset2retriever, learning_rate='2.5e-5', use_long_demonstration=False, target_task=None)

    # Instruct-RAG-ICL
    # in_context_ralm(model_size='8b', num_docs_test=5, dataset_name='ASQA', dataset2retriever=dataset2retriever, use_zero_shot_cot=False, use_short_demonstration=False, use_long_demonstration=True, data_split='test', max_instances=-1, num_device=-1, num_demo=-1, demo_seed=None, target_task='ASQA') # demo_seed = [13, 21, 42, 87, 100]
    