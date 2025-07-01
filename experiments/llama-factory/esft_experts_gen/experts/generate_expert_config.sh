python generate_expert_config.py \
    --eval_dataset=law \
    --expert_scores_dir=../law/expert_eval_results \
    --output_path=../law/expert_configs_token.json \
    --score_function=token \
    --top_p=0.4 \
    --token_threshold=0.01


# score_functions: gate token