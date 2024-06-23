# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import json
import dataclasses
from tqdm import tqdm
from functools import partial
from typing import Dict, Sequence, Union

import torch
import numpy as np
import transformers
import log_utils, common_utils 

IGNORE_INDEX = -100
logger = log_utils.get_logger(__name__)


class SFTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_list: list[dict],
        prompt_dict: dict,
        tokenizer: transformers.PreTrainedTokenizer,
        n_docs: int,
    ):
        super(SFTDataset, self).__init__()

        sft_data = preprocess_for_rag(data_list=data_list, prompt_dict=prompt_dict, tokenizer=tokenizer, n_docs=n_docs)

        self.input_ids = sft_data["input_ids"]
        self.labels = sft_data["labels"]

        self.metadata = sft_data["metadata"]
        self.tokenization_metadata = sft_data["tokenization_metadata"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclasses.dataclass
class DataCollatorForSFTDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

def make_supervised_data(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
):
    
    prompt_dict = common_utils.jload(data_args.prompt_dict_path)

    data_path = os.path.join('dataset', data_args.dataset_name, 'train.json')
    logger.warning(f"Loading training set from: {data_path}")
    data_list = common_utils.jload(data_path)

    train_dataset = SFTDataset(
        data_list=data_list,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        n_docs=data_args.n_docs,
    )

    data_collator = DataCollatorForSFTDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, data_collator=data_collator)


def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"
    if question.startswith("."):
        question = question.lstrip(". ")

    return question[0].lower() + question[1:]


def build_contexts(example, n_docs):

    if len(example["ctxs"]) > 0 and example["ctxs"][0]["score"] > example["ctxs"][1]["score"]:
        ctxs_list = example["ctxs"][:n_docs][::-1]
    else:
        ctxs_list = example["ctxs"][:n_docs]

    docs_text = "\n\n".join([f"Document {idx+1} (Title: {ctx['title']}): {ctx['text']}" for idx, ctx in enumerate(ctxs_list)])
    doc_prompt = f"{docs_text}\n\n"
    
    return doc_prompt


def preprocess_for_rag(
    data_list: list[dict],
    prompt_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    n_docs: int,
    verbose=True,
) -> dict[str, Union[torch.Tensor, Sequence[torch.Tensor]]]:
    """Preprocess the data by tokenizing."""

    sources = []
    targets = []

    assistant_prefix = prompt_dict['assistant_prefix']
    assist_prefix_len = len(tokenizer.encode(assistant_prefix, add_special_tokens=False, return_tensors="pt")[0])

    user_prefix = prompt_dict['user_prefix']
    user_prefix_id = tokenizer.encode(user_prefix, add_special_tokens=True, return_tensors="pt")[0]
    user_prefix_len = len(user_prefix_id)

    for sample in data_list:
        query_prompt = prompt_dict['query_prompt'] + normalize_question(sample['question'])
        doc_prompt = build_contexts(sample, n_docs=n_docs)
        sources.append(doc_prompt + query_prompt)
    
        target_prompt = assistant_prefix + sample['rationale'] + tokenizer.eos_token
        targets.append(target_prompt)

    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized = _tokenize_fn(examples, tokenizer, max_len_offset = [user_prefix_len] * len(examples), add_special_tokens=False)

    input_ids = [torch.cat([user_prefix_id, ctx]) for ctx in examples_tokenized["input_ids"]]
    targets_tokenized = _tokenize_fn(targets, tokenizer, add_special_tokens=False)

    labels = copy.deepcopy(input_ids)

    for idx, label in enumerate(labels):
        target_len = len(targets_tokenized["input_ids"][idx])            
        
        if idx == 0:
            logger.warning(f'\n===DEBUG Input:\n{json.dumps(tokenizer.decode(label))}===')
            logger.warning(f'\n===DEBUG Target:\n{label[-(target_len - assist_prefix_len):]} ==> {json.dumps(tokenizer.decode(label[-(target_len - assist_prefix_len):]))}===')

        assert torch.all(labels[idx][-(target_len-assist_prefix_len):].eq(targets_tokenized["input_ids"][idx][assist_prefix_len:])) 

        label[:-(target_len - assist_prefix_len)] = IGNORE_INDEX 

    packaged_data = dict(
        input_ids=input_ids,
        labels=labels,
        metadata=dict(),
        tokenization_metadata=examples_tokenized["tokenization_metadata"],
    )

    if verbose:
        logger.warning(f"Tokenization metadata:\n{json.dumps(packaged_data['tokenization_metadata'])}")

    return packaged_data

    
def _tokenize_text(x, tokenizer, padding, add_special_tokens):
    tokenized = tokenizer(
        text=x,
        return_tensors="pt",
        padding=padding,
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=add_special_tokens,
    )
    return tokenized

def _tokenize_text_with_offset(x, tokenizer, padding, add_special_tokens):
    tokenized = tokenizer(
        text=x[0],
        return_tensors="pt",
        padding=padding,
        max_length=tokenizer.model_max_length - x[1],
        truncation=True,
        add_special_tokens=add_special_tokens,
    )
    return tokenized

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, max_len_offset=None, add_special_tokens=True) -> dict:
    """Tokenize a list of strings and return the tokenized content"""
    padding = getattr(tokenizer, "padding", "longest")
    if max_len_offset is not None:
        tokenized_list = list(
            map(
                partial(_tokenize_text_with_offset, tokenizer=tokenizer, padding=padding, add_special_tokens=add_special_tokens),
                zip(strings, max_len_offset),
            )
        )
    else:
        tokenized_list = list(
            map(
                partial(_tokenize_text, tokenizer=tokenizer, padding=padding, add_special_tokens=add_special_tokens),
                strings,
            )
        )

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
        tokenization_metadata=dict(
            num_examples=len(tokenized_list),
            input_ids_avg_len=np.mean(input_ids_lens),
            input_ids_max_len=max(input_ids_lens),
            input_ids_min_len=min(input_ids_lens),
            labels_avg_len=np.mean(labels_lens),
            labels_max_len=max(labels_lens),
            labels_min_len=min(labels_lens),
            model_max_length=tokenizer.model_max_length,
        ),
    )

# Inference Data Utils

def format_prompt(
        dataset_name: str,
        example: dict, 
        n_docs: int,
        prompt_dict: dict,
        tokenizer: transformers.PreTrainedTokenizer,
        do_rationale_generation: bool,
        demos: list = [],
        ) -> str:
    """Formats a prompt with a prompt_dict formatter.

    Args:
        example: A dict-like object with required keys "instruction" and "input"
        prompt_dict: Dictionary containing the keys "prompt_noinputs" and "prompt_inputs" which have
            placeholders corresponding to the keys from `example`. E.g. "{instruction}".

    Returns:
        A formatted prompt string.

    Examples
    --------
    >>> format_prompt(dict(instruction="test", input=""), prompt_dict=dict(prompt_noinputs="prompt {instruction} "))
    "prompt test"
    """
    example['question'] = normalize_question(example['question'])
    max_length = tokenizer.model_max_length

    query_prompt = prompt_dict['query_prompt'].format_map(example)
    target_prefix = ""

    doc_prompt = build_contexts(example, n_docs=n_docs)

    prefix = prompt_dict['user_prefix']

    if do_rationale_generation:
        query_prompt = ''
        prefix += prompt_dict['demo_prefix'].format_map(example)
        target_prefix += prompt_dict['rationale_generation_instruction'].format_map(example) + prompt_dict['rationale_generation_postfix_' + dataset_name]

    elif len(demos) > 0:
        prefix += prompt_dict['demo_task_instruction']

        for idx, demo in enumerate(demos):
            demo_question = normalize_question(demo['question'])
            demo_rationale = demo['rationale']
            prefix += f"###\n\nExample {idx+1}\n\nQuestion: {demo_question}\n\nAnswer: {demo_rationale}\n\n"

        prefix += prompt_dict['demo_postfix']

    prefix_tokenized_id = tokenizer(prefix, return_tensors="pt", add_special_tokens=True).input_ids
    prefix_len = len(prefix_tokenized_id)

    target_prefix += prompt_dict['assistant_prefix']

    input_ids = tokenizer(doc_prompt + query_prompt + target_prefix, return_tensors="pt", add_special_tokens=False).input_ids

    if input_ids.shape[-1] > max_length - prefix_len:
        input_ids = input_ids[..., -(max_length - prefix_len):]
    input_ids = torch.cat([prefix_tokenized_id, input_ids], axis=-1)
    
    formatted_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    return formatted_prompt

def format_prompt_with_data_list(
    data_list: list[dict],
    dataset_name: str,
    prompt_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    n_docs: int = 5,
    demos: list = [],
    do_rationale_generation: bool = False,
):

    data = copy.deepcopy(data_list)
    logger.warning(f"Formatting prompts...")
    formatted_data = [format_prompt(dataset_name, example, n_docs, prompt_dict, tokenizer, do_rationale_generation, demos) for example in tqdm(data)]

    return formatted_data