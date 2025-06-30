import torch
from functools import partial
from datasets import load_dataset

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


def get_train_val_dataset(
    do_eval=False,
    train_file=None,
    test_file=None,
    dataset_name=None,
    dataset_args={}
):
    if dataset_name is not None:
        raw_train_dataset = load_dataset(dataset_name, data_files=train_file)["train"]

        if do_eval:
            if test_file is not None:
                raw_val_dataset = load_dataset(dataset_name, data_files=test_file)["train"]
            else:
                # Split the training set into train and validation (90/10)
                split_datasets = raw_train_dataset.train_test_split(test_size=0.1)
                raw_train_dataset = split_datasets["train"]
                raw_val_dataset = split_datasets["test"]
        else:
            raw_val_dataset = None

    else:
        if train_file is not None:
            raw_train_dataset = load_dataset(
                "json",
                data_files=train_file,
                **dataset_args,
            )["train"]
        else:
            raise ValueError("Training file must be provided.")

        # Load validation dataset (optional)
        if do_eval:
            if test_file is not None:
                raw_val_dataset = load_dataset(
                    "json",
                    data_files=test_file,
                    **dataset_args,
                )["train"]
            else:
                # Split the training set into train and validation (90/10)
                split_datasets = raw_train_dataset.train_test_split(test_size=0.1)
                raw_train_dataset = split_datasets["train"]
                raw_val_dataset = split_datasets["test"]
        else:
            raw_val_dataset = None

    return raw_train_dataset, raw_val_dataset
