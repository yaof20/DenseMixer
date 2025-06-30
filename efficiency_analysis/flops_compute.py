#!/usr/bin/env python3

import json
import argparse

def model_training_cost_analysis(model_config_path):
    config = json.load(open(model_config_path))
    
    vocab_size = config["vocab_size"]
    hidden_size = config["hidden_size"]
    max_position_embeddings = config["max_position_embeddings"]
    num_hidden_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    
    dense_layer_count = config.get("first_k_dense_replace", 0)
    moe_layer_count = num_hidden_layers - dense_layer_count
    
    intermediate_size = config["intermediate_size"]          # for dense layers
    moe_intermediate_size = config["moe_intermediate_size"]  # for MoE layers
    
    n_routed_experts = config["n_routed_experts"] 
    num_experts_per_tok = config["num_experts_per_tok"]
    
    # ----- Parameter Count -----
    params_emb = vocab_size * hidden_size * 2
    params_attn = hidden_size * hidden_size * 4  # for q, k, v, o
    params_ln = hidden_size * 2
    
    params_mlp_dense = hidden_size * intermediate_size * 3
    
    gating_params = hidden_size * n_routed_experts
    expert_params = n_routed_experts * (3 * hidden_size * moe_intermediate_size)
    
    dense_layer_params = params_attn + params_ln + params_mlp_dense
    moe_layer_params = params_attn + params_ln + gating_params + expert_params
    
    # Total parameters across layers:
    total_params = dense_layer_count * dense_layer_params + moe_layer_count * moe_layer_params + params_emb

    # ----- FLOPs Analysis -----
    batch_size = 1
    seq_len = max_position_embeddings
    head_dim = hidden_size // num_heads
    
    qkv_proj_flops = 3 * (2 * batch_size * seq_len * hidden_size * hidden_size)
    rope_flops = batch_size * seq_len * num_heads * 3 * head_dim
    attn_score_flops = num_heads * (2 * batch_size * head_dim * seq_len * seq_len)
    attn_sftmx_flops = num_heads * (3 * batch_size * seq_len * seq_len)
    attn_weight_flops = batch_size * num_heads * seq_len * head_dim * 2 * seq_len
    attn_output_flops = batch_size * seq_len * hidden_size * 2 * hidden_size
    attn_residual_flops = batch_size * seq_len * hidden_size

    attn_total_flops = (qkv_proj_flops + rope_flops + attn_score_flops +
                        attn_sftmx_flops + attn_weight_flops + attn_output_flops +
                        attn_residual_flops)
    
    rms_flops = batch_size * seq_len * (4 * hidden_size + 2) * 2
    
    # Dense MLP FLOPs (for the first_k_dense layers)
    mlp_gate_flops_dense = batch_size * seq_len * intermediate_size * 2 * hidden_size
    mlp_up_flops_dense   = batch_size * seq_len * intermediate_size * 2 * hidden_size
    mlp_activation_flops_dense = 4 * batch_size * seq_len * intermediate_size
    mlp_element_wise_flops_dense = batch_size * seq_len * intermediate_size
    mlp_down_flops_dense = batch_size * seq_len * hidden_size * 2 * intermediate_size
    mlp_residual_flops_dense = batch_size * seq_len * hidden_size
    dense_mlp_flops = (mlp_gate_flops_dense + mlp_up_flops_dense + mlp_activation_flops_dense +
                       mlp_element_wise_flops_dense + mlp_down_flops_dense + mlp_residual_flops_dense)
    
    # MoE MLP FLOPs (for the remaining layers):
    gating_flops = 2 * batch_size * seq_len * hidden_size * n_routed_experts
    expert_flops = 3 * hidden_size * moe_intermediate_size * num_experts_per_tok * batch_size * seq_len
    moe_mlp_flops = gating_flops + expert_flops
    
    # For dense layers, total FLOPs per layer:
    dense_layer_flops = attn_total_flops + dense_mlp_flops + rms_flops
    # For MoE layers, total FLOPs per layer:
    moe_layer_flops = attn_total_flops + moe_mlp_flops + rms_flops
    
    # Compute average FLOPs per layer weighted by the layer counts:
    total_flops_layers = dense_layer_count * dense_layer_flops + moe_layer_count * moe_layer_flops
    avg_layer_flops = total_flops_layers / num_hidden_layers
    flops_layer_TF = avg_layer_flops / (10**12)  # Convert to TFLOPs

    # ----- Memory Analysis -----
    bytes_per_fp16 = 2
    # Parameters
    mem_for_params = (dense_layer_count * dense_layer_params + moe_layer_count * moe_layer_params +
                      params_emb) * bytes_per_fp16

    # Activations
    mem_for_input = batch_size * seq_len * hidden_size * bytes_per_fp16
    mem_for_qkv = 3 * batch_size * seq_len * hidden_size * bytes_per_fp16
    mem_for_attn_scores = batch_size * num_heads * seq_len * seq_len * bytes_per_fp16
    mem_for_attn_output = batch_size * seq_len * hidden_size * bytes_per_fp16
    # MLP activations: use dense MLP activation size for dense layers and MoE activation size for MoE layers, then take a weighted average.
    mem_for_mlp_act_dense = batch_size * seq_len * intermediate_size * bytes_per_fp16
    mem_for_mlp_act_moe   = batch_size * seq_len * moe_intermediate_size * num_experts_per_tok * bytes_per_fp16
    avg_mem_for_mlp_act = ((dense_layer_count * mem_for_mlp_act_dense +
                            moe_layer_count * mem_for_mlp_act_moe) / num_hidden_layers)
    
    mem_for_activations = (mem_for_input + mem_for_qkv +
                           mem_for_attn_scores + mem_for_attn_output +
                           avg_mem_for_mlp_act)

    peak_mem = mem_for_params + mem_for_activations
    peak_memory_GB = peak_mem / (1024**3)
    
    return total_params, flops_layer_TF, peak_memory_GB



def main():
    # Run the deepseek analysis function
    num_parameters, num_flops_forward, memory_cost = model_training_cost_analysis("qwen3_config.json")
    num_parameters_dm, num_flops_forward_dm, memory_cost_dm = model_training_cost_analysis("qwen3_densemixer_config.json")

    
    num_flops_backward = num_flops_forward * 2
    num_flops = num_flops_forward + num_flops_backward

    num_flops_backward_dm = num_flops_forward * 2  # densemixer's backward is unchanged (almost)
    num_flops_dm = num_flops_forward_dm + num_flops_backward_dm
    
    print(f"Model Training Cost Analysis Results --- Conventional Training for Qwen3-30B-A3B ---")
    print(f"Number of parameters: {num_parameters:,}")
    print(f"Number of Forward TFLOPs per layer: {num_flops_forward:.2f}")
    print(f"Number of Backward TFLOPs per layer: {num_flops_backward:.2f}")
    print(f"Number of TFLOPs per layer: {num_flops:.2f}")
    print(f"Peak memory cost: {memory_cost:.2f} GBs")

    print(f"\n\nModel Training Cost Analysis Results: --- Dense Mixer Training for Qwen3-30B-A3B ---")
    print(f"Number of parameters: {num_parameters_dm:,}")
    print(f"Number of Forward TFLOPs per layer: {num_flops_forward_dm:.2f}")
    print(f"Number of Backward TFLOPs per layer: {num_flops_backward_dm:.2f}")
    print(f"Number of TFLOPs per layer: {num_flops_dm:.2f}")
    print(f"Peak memory cost: {memory_cost_dm:.2f} GBs")

    print(f'\nFLOPs: DenseMixer / Conventional = {num_flops_dm / num_flops:.2f}x')

if __name__ == "__main__":
    main() 