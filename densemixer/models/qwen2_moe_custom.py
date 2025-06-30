import torch
import torch.nn.functional as F
from ..logging_utils import log_custom_forward_usage

class CustomQwen2MoeSparseMoeBlock:
    @staticmethod
    def forward(self, hidden_states: torch.Tensor):
        """
        Inherited from official OlmoeSparseMoeBlock, implements dense backward functionality:
        Forward output remains the same as official (i.e., sparse computation results),
        but during backward propagation, dense computation gradients are passed back through straight-through gradient,
        dense output is obtained by computing each expert on all tokens and weighted by full routing weights.

        Input:
            hidden_states: Tensor, shape (batch_size, sequence_length, hidden_dim)
        Output:
            final_output: Tensor, shape (batch_size, sequence_length, hidden_dim)
            router_logits: Tensor, shape (batch_size * sequence_length, num_experts)
        """
        log_custom_forward_usage("Qwen2-MoE")
        batch_size, seq_length, hidden_dim = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device

        flat_hidden = hidden_states.view(-1, hidden_dim)  # (B*seq_len, hidden_dim)
        N_tokens = flat_hidden.size(0)

        # Compute routing logic
        router_logits = self.gate(flat_hidden)  # (B*L, num_experts)
        router_logits = router_logits.to(dtype=dtype)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)  # (B*L, num_experts)

        # Select top-k experts and cast to match input dtype to avoid dtype mismatch on in-place updates
        routing_weights_topk, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights_topk = routing_weights_topk.to(dtype=dtype)
        if self.norm_topk_prob:
            routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(dim=-1, keepdim=True)
            routing_weights_topk = routing_weights_topk.to(dtype=dtype)

        # Convert full routing_weights to consistent dtype for dense accumulation
        routing_weights = routing_weights.to(dtype=dtype)
        # Add shared expert contribution to both sparse and dense outputs
        shared_expert_output = self.shared_expert(flat_hidden)  # (N_tokens, hidden_dim)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(flat_hidden)) * shared_expert_output
        # Prepare accumulators: one for dense_outputs, one for sparse_outputs
        dense_outputs = torch.zeros((N_tokens, hidden_dim), dtype=dtype, device=device)
        sparse_outputs = torch.zeros((N_tokens, hidden_dim), dtype=dtype, device=device)

        # For mapping top-k positions when accumulating sparse_outputs
        # selected_experts: (N_tokens, top_k)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # Compute current expert output for all tokens
            expert_output = expert_layer(flat_hidden)  # (N_tokens, hidden_dim)
            # Register hook for all experts to mask non-selected token gradients
            activation_mask = (selected_experts == expert_idx).any(dim=1).float().unsqueeze(-1).to(dtype)
            if expert_output.requires_grad:
                expert_output.register_hook(lambda grad, mask=activation_mask: grad * mask)
            expert_output = expert_output.to(dtype=dtype)

            # Dense accumulation: multiply by full routing weight and add
            weight_full = routing_weights[:, expert_idx].unsqueeze(-1)  # (N_tokens, 1)
            dense_outputs = dense_outputs + expert_output * weight_full

            # Sparse accumulation: find tokens where this expert is among top_k
            matches = (selected_experts == expert_idx)
            if matches.any():
                token_indices, k_indices = torch.where(matches)
                weights_topk = routing_weights_topk[token_indices, k_indices].unsqueeze(-1)  # (num_matches, 1)
                sparse_outputs[token_indices] = sparse_outputs[token_indices] + expert_output[token_indices] * weights_topk

        sparse_outputs = sparse_outputs + shared_expert_output
        dense_outputs = dense_outputs + shared_expert_output

        # Combine sparse forward output and dense backward output
        final_flat = sparse_outputs.detach() + (dense_outputs - dense_outputs.detach())
        final_flat = final_flat.to(dtype=dtype)
        final_output = final_flat.view(batch_size, seq_length, hidden_dim)

        return final_output, router_logits