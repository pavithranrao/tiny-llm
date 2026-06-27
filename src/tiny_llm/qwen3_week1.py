import torch
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rms_norm_eps = rms_norm_eps
        self.rope = RoPE(
            self.head_dim,
            self.max_seq_len,
            self.theta,
        )

    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        H_q = self.num_heads
        H = self.num_kv_heads
        D = self.head_dim

        q = linear(x, self.wq).reshape(B, L, H_q, D)
        k = linear(x, self.wk).reshape(B, L, H, D)
        v = linear(x, self.wv).reshape(B, L, H, D)

        projection_q = torch.nn.functional.rms_norm(q, self.q_norm.shape, self.q_norm)
        projection_k = torch.nn.functional.rms_norm(k, self.k_norm.shape, self.k_norm)

        projection_q = self.rope(projection_q, offset=slice(0, L))
        projection_k = self.rope(projection_k, offset=slice(0, L))

        projection_q = projection_q.permute(0, 2, 1, 3)
        projection_k = projection_k.permute(0, 2, 1, 3)
        projection_v = v.permute(0, 2, 1, 3)

        x = scaled_dot_product_attention_grouped(
            query=projection_q,
            key=projection_k,
            value=projection_v,
            mask=mask,
        )
        x = x.permute(0, 2, 1, 3).reshape(B, L, H_q * D)
        return linear(x, self.wo)


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        w_input_layernorm: torch.Tensor,
        w_post_attention_layernorm: torch.Tensor,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.w_input_layernorm = w_input_layernorm
        self.w_post_attention_layernorm = w_post_attention_layernorm
        self.max_seq_len = max_seq_len
        self.theta = theta

    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        pass


class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        self.mlx_model = mlx_model

    def __call__(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        pass
