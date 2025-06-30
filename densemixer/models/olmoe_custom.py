import torch
import torch.nn.functional as F
from ..logging_utils import log_custom_forward_usage

class CustomOlmoeSparseMoeBlock:
    @staticmethod
    def forward(self, hidden_states: torch.Tensor):
        """
        forward_partscale_fixep_norm_dtch
        """
        log_custom_forward_usage("OLMoE")
        batch_size, seq_length, hidden_dim = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device

        flat_hidden = hidden_states.view(-1, hidden_dim)  # (B*seq_len, hidden_dim)
        N_tokens = flat_hidden.size(0)

        # Compute routing logic
        router_logits = self.gate(flat_hidden).to(dtype=dtype)  # (B*L, num_experts)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)  # (B*L, num_experts)

        # Select top-k experts
        routing_weights_topk, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            norm_ratio = routing_weights_topk.sum(dim=-1, keepdim=True)
            # Normalize top-k routing weights
            routing_weights_topk = routing_weights_topk / norm_ratio
            # Only scale the selected top-k positions in routing_weights
            mask = F.one_hot(selected_experts, num_classes=self.num_experts).sum(dim=1).to(dtype)
            # ------------------------------------Choose Section-----------------------------------------------
            # current --> partscale_fix_expert implementation
            routing_weights = routing_weights * (1.0 - mask) / norm_ratio.detach() + routing_weights * mask / norm_ratio

            # should be --> the gated implemenation, by comment out the line above and uncomment the two lines below
            # gated = routing_weights.detach() * mask + (routing_weights - routing_weights.detach())
            # routing_weights = gated / gated.sum(dim=-1, keepdim=True)
            # ------------------------------------Choose Section-----------------------------------------------

        routing_weights_topk = routing_weights_topk.to(dtype=dtype)

        # Convert full routing_weights to consistent dtype for dense accumulation
        routing_weights = routing_weights.to(dtype=dtype)

        # Prepare accumulators: one for dense_outputs, one for sparse_outputs
        dense_outputs = torch.zeros((N_tokens, hidden_dim), dtype=dtype, device=device)
        sparse_outputs = torch.zeros((N_tokens, hidden_dim), dtype=dtype, device=device)

        # For mapping top-k positions when accumulating sparse_outputs
        # selected_experts: (N_tokens, top_k)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # Compute current expert output for all tokens
            expert_output = expert_layer(flat_hidden).to(dtype=dtype)  # (N_tokens, hidden_dim)
            activation_mask = (selected_experts == expert_idx).any(dim=1).float().unsqueeze(-1).to(dtype)
            if expert_output.requires_grad:
                expert_output.register_hook(lambda grad, mask=activation_mask: grad * mask)
            expert_output = expert_output.to(dtype=dtype)
            # Dense accumulation: multiply by full routing weight and add
            weight_full = routing_weights[:, expert_idx].unsqueeze(-1)  # (N_tokens, 1)
            dense_outputs = dense_outputs + expert_output * weight_full

            # Sparse accumulation: find tokens where this expert is among top_k
            # matches: Boolean mask where selected_experts == expert_idx → shape (N_tokens, top_k)
            matches = (selected_experts == expert_idx)
            if matches.any():
                # locations: tuple of (token_indices, k_indices)
                token_indices, k_indices = torch.where(matches)
                # corresponding top-k weights
                weights_topk = routing_weights_topk[token_indices, k_indices].unsqueeze(-1)  # (num_matches, 1)
                # Accumulate sparse_outputs only for matched tokens
                sparse_outputs[token_indices] = sparse_outputs[token_indices] + expert_output[token_indices] * weights_topk

        # Combine sparse forward output and dense backward output
        final_flat = sparse_outputs.detach() + (dense_outputs - dense_outputs.detach())
        final_flat = final_flat.to(dtype=dtype)
        final_output = final_flat.view(batch_size, seq_length, hidden_dim)

        return final_output, router_logits