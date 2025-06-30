import json
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset
from benchmarks import *
import multiprocessing as mp

def load_model(base_model_path, cache_dir):
    # load pretrained model:
    if cache_dir != "":
        model, tokenizer = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=cache_dir), AutoTokenizer.from_pretrained(base_model_path, cache_dir=cache_dir)

        return model, tokenizer

    model, tokenizer = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16), AutoTokenizer.from_pretrained(base_model_path)

    return model, tokenizer



def main(args):
    config = {
        "max_new_tokens": args.max_new_tokens,
        "eval_batch_size": args.eval_batch_size,
        "openai_api_key": args.openai_api_key,
        "model_path": args.model_path,  # add this for vLLMWrapper
        "gpu_count": args.gpu_count
    }

    eval_datasets = args.eval_datasets.split(",")
    evaluator_map={"intent": IntentEvaluator, "summary": SummaryEvaluator, "law": LawEvaluator, "translation": TranslationEvaluator}

    for dataset_name in eval_datasets:
        print(f"Running evaluation on {dataset_name}...")

        dataset = load_dataset(f"RoxanneWsyw/ESFT-{dataset_name}",split="test")

        if args.debug:
            print("Debugging mode: Shortening dataset to 16 samples.")
            dataset = dataset[:4]
        

        evaluator = evaluator_map[dataset_name](dataset, config)

        results, metrics = evaluator.evaluate()

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{dataset_name}.jsonl")

        with open(output_path, "w", encoding="utf-8") as f:
            for res, m in zip(results, metrics):
                obj = {"example": res, "score": m}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Evaluation complete. Results saved to:", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--eval_datasets", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--openai_api_key", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--gpu_count", type=int, default=2)
    args = parser.parse_args()

    main(args)