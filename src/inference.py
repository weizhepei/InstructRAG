import os
import sys
import argparse
import data_utils
import common_utils
from metrics import get_metrics
from vllm import LLM, SamplingParams

def generate_rationale(args):
    data_path = f'dataset/{args.dataset_name}/train.json'
    print(f"Loading training set from: {data_path}")
    train_data = common_utils.jload(data_path)[:args.max_instances]

    llm = LLM(model=args.model_name_or_path, download_dir=args.cache_dir, max_model_len=args.max_tokens)

    tokenizer = llm.get_tokenizer()
    prompt_dict = common_utils.jload(args.prompt_dict_path)

    prompts = data_utils.format_prompt_with_data_list(
        data_list=train_data,
        dataset_name=args.dataset_name,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        n_docs=args.n_docs,
        do_rationale_generation=True,
    )

    sampling_params = SamplingParams(temperature=args.temperature, 
                                    max_tokens=args.max_tokens, 
                                    seed=args.seed,
                                    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    
    outputs = llm.generate(prompts, sampling_params)

    output_file = os.path.join(args.output_dir, "with_rationale/train.json")

    save_outputs(outputs, train_data, output_file, args.n_docs)
   
def eval_model(args):
    data_path = f'dataset/{args.dataset_name}/test.json'
    print(f"Loading eval set from: {data_path}")
    test_data = common_utils.jload(data_path)[:args.max_instances]

    print(f'Loading model {args.rag_model}...')
    if args.rag_model == 'InstructRAG-FT':
        demos = []
        if args.load_local_model:
            llm = LLM(model=f'saved_checkpoints/InstructRAG-FT/{args.dataset_name}',  max_model_len=args.max_tokens)
        else:
            llm = LLM(model=f'meng-lab/{args.dataset_name}-InstructRAG-FT', download_dir=args.cache_dir, max_model_len=args.max_tokens)
    elif args.rag_model == 'InstructRAG-ICL':
        demos = common_utils.jload(f'dataset/{args.dataset_name}/demos.json')
        llm = LLM(model='meta-llama/Meta-Llama-3-8B-Instruct', download_dir=args.cache_dir, max_model_len=args.max_tokens)

    tokenizer = llm.get_tokenizer()
    prompt_dict = common_utils.jload(args.prompt_dict_path)
 
    prompts = data_utils.format_prompt_with_data_list(
        data_list=test_data,
        dataset_name=args.dataset_name,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        n_docs=args.n_docs,
        demos=demos,
    )
    
    sampling_params = SamplingParams(temperature=args.temperature, 
                                    max_tokens=args.max_tokens, 
                                    seed=args.seed,
                                    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    
    outputs = llm.generate(prompts, sampling_params)

    output_file = os.path.join(args.output_dir, "result.json")

    eval_results = save_outputs(outputs, test_data, output_file, args.n_docs)
    get_metrics(eval_results, args.output_dir, is_asqa=args.dataset_name == 'ASQA')

def save_outputs(outputs, test_data, output_file, n_docs):
    # Save the outputs as a JSON file.
    output_data = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        sample = test_data[i]
        output_data.append({
            "question": sample["question"],
            "answers": sample["answers"],
            "qa_pairs": sample["qa_pairs"] if "qa_pairs" in sample else None,
            "rationale": generated_text,
            "prompt": prompt,
            "ctxs": sample["ctxs"][:n_docs][::-1] if (sample["ctxs"][0]['score'] > sample["ctxs"][1]['score']) else sample["ctxs"][:n_docs],
            })
        
    common_utils.jdump(output_data, output_file)
    print(f"Outputs saved to {output_file}")

    return output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--rag_model', type=str, choices=['InstructRAG-FT', 'InstructRAG-ICL'], default='InstructRAG-FT', help='InstructRAG model: InstructRAG-FT or InstructRAG-ICL')
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='name of the model in Hugging Face model hub or path to the model')
    parser.add_argument('--load_local_model', action='store_true', help='Load local model')
    parser.add_argument('--do_rationale_generation', action='store_true', help='Generate rationales on training data')
    parser.add_argument('--n_docs', type=int, default=5, help='Number of retrieved documents')
    parser.add_argument('--output_dir', type=str, help='Path to the output file')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory to cached models')
    parser.add_argument('--prompt_dict_path', type=str, default="src/rag.json")
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for sampling')
    parser.add_argument('--max_tokens', type=int, default=4096, help='Maximum number of tokens')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_instances', type=int, default=sys.maxsize)

    args = parser.parse_args()

    if args.do_rationale_generation:
        generate_rationale(args)
    else:
        eval_model(args)
