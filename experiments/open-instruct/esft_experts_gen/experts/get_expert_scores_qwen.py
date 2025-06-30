import json
import os
import torch
import argparse
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.multiprocessing as mp
from itertools import accumulate
from accelerate import dispatch_model
from datasets import load_dataset
from functools import partial


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }

def encode_with_question_answers_format(example, tokenizer, max_seq_length):
    prompt = f'Question: {example["question"]}\nAnswer:'
    completion = ""
    if "answers" in example:
        completion = example["answers"][0]
    if "answer" in example:
        completion = example["answer"]
    example_text = prompt + completion
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=max_seq_length, truncation=True)
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def encode_with_multi_choices_format(example, tokenizer, max_seq_length):
    choices = example["choices"]
    prompt = f'Question: {example["question"].strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]} \nAnswer:'

    doc_to_choice = ["A", "B", "C", "D"]
    answer_id = int(example["answer"])
    completion = f"{doc_to_choice[answer_id]}. {choices[answer_id]}"
    example_text = prompt + completion
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }

def create_code_w_input_prompt(instruction, input):
    return f"""
        Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
    """

def create_code_prompt(instruction):
    return f"""
        Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Response:
    """


def encode_with_input_output_format(example, tokenizer, max_seq_length):
    input_string = example["input"]
    instruction = example["instruction"]
    output = example["output"]
    if input_string is not None and input_string != "":
        prompt = create_code_w_input_prompt(instruction, input_string)
    else:
        prompt = create_code_prompt(instruction)

    completion = output
    example_text = prompt + completion
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=max_seq_length, truncation=True)
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def get_encode_function(dataset_name, tokenizer, max_seq_length):
    if "trivia_qa" in dataset_name or "gsm" in dataset_name:
        train_encode_fn = partial(
                encode_with_prompt_completion_format,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
            )
        val_encode_fn = partial(
                encode_with_question_answers_format,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
            )
        return train_encode_fn, val_encode_fn
    if "MBPP" in dataset_name or "math" in dataset_name or "ESFT" in dataset_name or "alpaca" in dataset_name:
        encode_fn = partial(
                encode_with_prompt_completion_format,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
            )
        return encode_fn, encode_fn

    if "MMLU" in dataset_name:
        encode_fn = partial(
                encode_with_multi_choices_format,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
            )
        return encode_fn, encode_fn
    if "CodeAlpaca" in dataset_name:
        encode_fn = partial(
                encode_with_input_output_format,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
            )
        return encode_fn, encode_fn
    return None, None


def preprocess(args, tokenizer):
    print("Loading dataset...")
    raw_datasets = load_dataset(args.dataset_name, data_files=args.file)
 
    _, encode_function = get_encode_function(args.dataset_name, tokenizer, args.max_seq_length)
    
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="torch")
    lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())
    return lm_datasets


def infer_auto_device_map(model, pp_splits, visible_devices):
    assert len(pp_splits) == len(visible_devices)
    device_map = {
        "model.embed_tokens": 0,
        "model.norm": len(pp_splits) - 1,
        "lm_head": len(pp_splits) - 1
    }
    assert len(model.model.layers) == sum(pp_splits)
    pp_splits = [0, *list(accumulate(pp_splits))]
    for idx, (start, end) in enumerate(zip(pp_splits[:-1], pp_splits[1:])):
        for i in range(start, end):
            device_map.update({f"model.layers.{i}": idx})
    for k, v in device_map.items():
        device_map[k] = visible_devices[v]
    return device_map


def hook_moe_gate(layer_idx, rank, args):
    def hook(module, input, output):
        gate_logits = output  

        # normalization
        gate_probs = torch.nn.functional.softmax(gate_logits, dim=-1) 

        sorted_experts = torch.argsort(gate_probs, dim=-1, descending=True)
        sorted_weights = torch.gather(gate_probs, -1, sorted_experts)


        seq_len, num_experts = sorted_experts.shape

        expert_ids = sorted_experts.flatten().tolist()
        expert_weights = sorted_weights.flatten().tolist()

        expert_ids_str = "\t".join(map(str, expert_ids))
        expert_weights_str = "\t".join(map(lambda x: f"{x:.6f}", expert_weights))

        output_dir = os.path.join(args.output_dir, f"rank_{rank}")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"expert_weights_{layer_idx}.txt")

        with open(output_file, "a") as f:
            f.write(f"{expert_ids_str}\t\t{expert_weights_str}\n")
    return hook


def eval_expert(rank, args, model, dataset):
    try:
        print(f"Rank {rank} starting expert evaluation...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        visible_devices = list(range(rank * args.gpus_per_rank, (rank + 1) * args.gpus_per_rank))
        device_map = infer_auto_device_map(model, [24], visible_devices)
        model = dispatch_model(model, device_map)
        model.config.expert_log_dir = os.path.join(args.output_dir, f"rank_{rank}")
        os.makedirs(model.config.expert_log_dir, exist_ok=True)

        n_sample_tokens = args.n_sample_tokens // args.world_size
        done_tokens = 0

        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                layer.mlp.gate.register_forward_hook(hook_moe_gate(layer_idx, rank, args))

        for instance in dataset['train']:
            print("done_tokens: ", done_tokens)
            input_ids = torch.tensor(instance['input_ids']).unsqueeze(0).to(model.device)
            labels = torch.tensor(instance['labels']).unsqueeze(0).to(model.device)

            outputs = model(input_ids=input_ids, labels=labels, return_dict=True)

            done_tokens += len(input_ids[0])
            if done_tokens >= n_sample_tokens:
                break

        print(f"Rank {rank} evaluation completed.", flush=True)

    except Exception as e:
        print(f"Error in process {rank}: {e}", flush=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model with adapters on a specified dataset.")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--file", type=str, default="")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the evaluation results")
    parser.add_argument("--world_size", type=int, default=4, help="Number of processes to use for evaluation")
    parser.add_argument("--gpus_per_rank", type=int, default=2, help="Number of GPUs per process")
    parser.add_argument("--n_sample_tokens", type=int, required=True, help="Token to sample for expert evaluation")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length for tokenization")
    parser.add_argument("--preprocessing_num_workers", type=int, default=4, help="Number of workers for preprocessing")
    parser.add_argument("--overwrite_cache", action='store_true', help="Overwrite dataset cache")
    
    args = parser.parse_args()
    random.seed(5934875)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model.config.log_expert_weights = True

    print("Preprocessing dataset...")
    dataset = preprocess(args, tokenizer)
    
    print("Start Evaluating...")
    mp.spawn(eval_expert, args=(args, model, dataset), nprocs=args.world_size, join=True)
