import torch
import torch.nn.functional as F
from ..logging_utils import log_custom_forward_usage

from torch.autograd import Function


class GateGradSTE(Function):
    """
    forward : Returns sparse MoE output (sparse_out) —— for inference and upstream computation  
    backward:  
        1. Directly pass grad_output back to sparse branch (maintain gradients for activated experts)  
        2. Recompute all expert outputs out_e_const under torch.no_grad()  
           dL/dw_{t,e} = grad_output_t · out_e_const_t  —— same as original dense MoE  
           This way gate can get complete gradients, but expert parameters won't build graph or backpropagate gradients
    """
    @staticmethod
    def forward(ctx,
                sparse_out: torch.Tensor,            # (N, H) requried gradient
                routing_weights: torch.Tensor,       # (N, E) requried gradient
                flat_hidden: torch.Tensor,           # (N, H) no gradient
                experts):                            # list[nn.Module]
        ctx.save_for_backward(routing_weights, flat_hidden)
        ctx.experts = experts                       # python objects can be stored in ctx
        return sparse_out

    @staticmethod
    def backward(ctx, grad_output):
        routing_weights, flat_hidden = ctx.saved_tensors
        experts = ctx.experts                       # list[Qwen3MoeMLP]
        N, E = routing_weights.shape
        H = grad_output.size(-1)

        # ---------------- gate gradients ----------------
        grad_routing = torch.zeros_like(routing_weights)  # (N, E)
        with torch.no_grad():                             # don't build computation graph
            for e, expert in enumerate(experts):
                out_e = expert(flat_hidden)               # (N, H) detached
                # gradient of each token w.r.t w_{t,e} = grad_out_t · out_e_t
                grad_routing[:, e] = torch.sum(grad_output * out_e, dim=-1)

        # ---------------- return ----------------
        # gradient w.r.t sparse_out = grad_output as is
        # gradient w.r.t routing_weights = grad_routing
        # flat_hidden and experts don't need gradients (None)
        return grad_output, grad_routing, None, None



class CustomQwen3MoeSparseMoeBlock:
    @staticmethod
    def forward(self, hidden_states: torch.Tensor):
        """
        forward_autograd_v1
        Accelerated Sparse-MoE forward:
        • Only compute gradients for activated (token, expert) pairs → memory/speed consistent with sparse  
        • Use GateGradSTE to recompute dense constants in backward phase, ensuring gate gradients are identical to original dense implementation  
        • Includes partscale_fix_expert normalization (when norm_topk_prob=True)
        """
        # -------------------------------------------------- shape & dtype --------------------------------------------------
        B, L, H = hidden_states.shape
        flat_hidden = hidden_states.view(-1, H)                # (N_tokens, H)
        N, E = flat_hidden.size(0), self.num_experts
        dtype, device = hidden_states.dtype, hidden_states.device

        # ------------------------------------------------ 1. routing ----------------------------------------------------------
        router_logits   = self.gate(flat_hidden)               # (N, E)
        routing_weights = torch.softmax(router_logits, dim=-1) # (N, E)   (float32 default consistent with PyTorch softmax)

        # ------------------------------------------------ 2. top-k ---------------------------------------------------------
        routing_weights_topk, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1                 # (N, k)
        )                                                      # (N,k), (N,k)

        # --------------------------- partscale_fix_expert normalization (optional) ---------------------------
        if self.norm_topk_prob:
            norm_ratio = routing_weights_topk.sum(dim=-1, keepdim=True)          # (N,1)
            routing_weights_topk = routing_weights_topk / norm_ratio             # normalized top-k

            mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=E
            ).sum(dim=1).to(routing_weights.dtype)                               # (N,E) 0/1

            # partscale_fix_expert: non-top-k weights decay proportionally, top-k re-normalized
            routing_weights = (
                routing_weights * (1.0 - mask) / norm_ratio.detach() +           # non-top-k
                routing_weights * mask / norm_ratio                              # top-k
            )

        # Convert to model-consistent dtype (needed for fp16/bf16)
        routing_weights      = routing_weights.to(dtype)
        routing_weights_topk = routing_weights_topk.to(dtype)

        # ------------------------------------------------ 3. sparse branch ------------------------------------------------------
        sparse_out = torch.zeros(N, H, device=device, dtype=dtype)

        for e in range(E):
            tok_idx, k_idx = torch.where(selected_experts == e)   # activated tokens
            if tok_idx.numel() == 0:
                continue

            # (n_sel, H) → with gradients
            out_grad = self.experts[e](flat_hidden.index_select(0, tok_idx))

            # ★★ Key: DETACH weights to block gate gradients ★★
            w_const = routing_weights_topk[tok_idx, k_idx].detach().unsqueeze(-1)  # (n_sel,1)
            sparse_out.index_add_(0, tok_idx, out_grad * w_const)

        # ------------------------------------------------ 4. straight-through estimator -----------------------------------------------------
        #   GateGradSTE is responsible for:
        #   • forward: directly return sparse_out
        #   • backward: recompute all expert outputs under no_grad for gate gradients
        final_flat = GateGradSTE.apply(
            sparse_out,              # (N,H)  → expert parameters need gradients
            routing_weights,         # (N,E)  → gate parameters need gradients
            flat_hidden.detach(),    # (N,H)  → constant
            self.experts             # python list
        )

        return final_flat.view(B, L, H), router_logits
    
    @staticmethod
    def forward_old(self, hidden_states: torch.Tensor):
        """
        original implementation with extra computation on the dense output
        """
        log_custom_forward_usage("Qwen3-MoE")
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