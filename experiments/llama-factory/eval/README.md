## How to evaluate different datasets

- gsm: `run_eval_gsm.sh`
Simply fill in the arguments in the shell script. Only a library installation is required.
```bash
pip install evaluate
```
- math: `lm_eval_math.sh`
You need to install `lm-eval`.
```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```
- MBPP and HumanEval: `lm_eval_code.sh`
The same as the pre-requirements of math evaluation.

- ESFT-intent, ESFT-law, ESFT-translation, ESFT-summary: `eval_esft.sh`
You need to fill in your own OpenAI key and modify `eval_datasets=intent,law,summary,translation`.

## Evaluation of Qwen3-30B-A3B:
- AIME24, Math-500, MBPP, Humaneval, gpqa_diamond, livecodebench: we use [Reasoning360](https://github.com/LLM360/Reasoning360) to complete the evaluation
- The configs are shown in ```eval/Reasoning360```

