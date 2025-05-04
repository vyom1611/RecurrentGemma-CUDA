# optimized_recurrent_gemma.py
import itertools
import math
from typing import Optional, Tuple, Dict, Literal, overload, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import os
from torch.utils.cpp_extension import load
from .modules import Embedder, ResidualBlock, MLPBlock

import itertools

class GriffinConfig:
    def __init__(
        self,
        vocab_size: int,
        width: int,
        mlp_expanded_width: int,
        recurrent_num_heads: int,
        attention_num_heads: int,
        lru_width: int,
        conv1d_temporal_width: int,
        final_w_init_variance_scale: float,
        block_types: tuple,
        attention_window_size: int,
        logits_soft_cap: float,
        embeddings_scale_by_sqrt_dim: bool,
    ):
        self.vocab_size = vocab_size
        self.width = width
        self.mlp_expanded_width = mlp_expanded_width
        self.recurrent_num_heads = recurrent_num_heads
        self.attention_num_heads = attention_num_heads
        self.lru_width = lru_width
        self.conv1d_temporal_width = conv1d_temporal_width
        self.final_w_init_variance_scale = final_w_init_variance_scale
        self.block_types = block_types
        self.attention_window_size = attention_window_size
        self.logits_soft_cap = logits_soft_cap
        self.embeddings_scale_by_sqrt_dim = embeddings_scale_by_sqrt_dim
        self.num_layers = len(block_types)

    @classmethod
    def from_preset(cls, vocab_size: int, preset: str) -> "GriffinConfig":
        presets = {
            "RECURRENT_GEMMA_2B_V1": {
                "width": 2560,
                "mlp_expanded_width": 7680,
                "recurrent_num_heads": 10,  
                "attention_num_heads": 40, 
                "lru_width": 2560,
                "conv1d_temporal_width": 4,
                "final_w_init_variance_scale": 2.0 / 26,
                "block_types": tuple(itertools.islice(itertools.cycle(["recurrent", "recurrent", "attention"]), 26)),
                "attention_window_size": 2048,
                "logits_soft_cap": 30.0,
                "embeddings_scale_by_sqrt_dim": True,
            }
        }
        if preset not in presets:
            raise ValueError("Unknown preset: " + preset)
        return cls(vocab_size=vocab_size, **presets[preset])


cuda_kernels = None
_cuda_file = os.path.join(os.path.dirname(__file__), "cuda_kernels.cu")
try:
    cuda_kernels = load(name="cuda_kernels", sources=[_cuda_file], verbose=True)
    HAS_CUDA_KERNELS = True
except Exception as e:
    print("WARNING: CUDA kernels could not be loaded, falling back to CPU implementations.")
    HAS_CUDA_KERNELS = False

attention_kernels = None
_attention_file = os.path.join(os.path.dirname(__file__), "attention.cu")
try:
    attention_kernels = load(name="attention", sources=[_attention_file], verbose=True)
    HAS_ATTENTION_KERNELS = True
except Exception as e:
    print("WARNING: Attention kernels could not be loaded, falling back to CPU attention.")
    HAS_ATTENTION_KERNELS = False

def rnn_scan(x: torch.Tensor, 
             a: torch.Tensor, 
             reset: torch.Tensor, 
             h0: Optional[torch.Tensor] = None):
    """
    CUDA-accelerated RNN scan operation.
    
    Args:
        x: Input tensor of shape [batch, seq_len, features]
        a: Recurrence weights of shape [batch, seq_len, features]
        reset: Reset flags of shape [batch, seq_len] (bool)
        h0: Initial hidden state of shape [batch, features] or None
        
    Returns:
        y: Output sequence [batch, seq_len, features]
        last_h: Final hidden state [batch, features]
    """
    assert x.device == a.device == reset.device
    if h0 is not None:
        assert h0.device == x.device
    
    assert x.dtype == a.dtype
    if h0 is not None:
        assert h0.dtype == x.dtype
    
    # Convert to float32 if using BFloat16 for more stable computation
    orig_dtype = x.dtype
    needs_conversion = orig_dtype == torch.bfloat16
    
    if needs_conversion:
        x = x.float()
        a = a.float()
        if h0 is not None:
            h0 = h0.float()
    
    # Call CUDA kernel
    empty_tensor = torch.tensor([], device=x.device)
    h0_tensor = h0 if h0 is not None else empty_tensor
    y, last_h = cuda_kernels.rnn_scan_cuda_forward(x, a, reset, h0_tensor)
    
    if needs_conversion:
        y = y.to(orig_dtype)
        last_h = last_h.to(orig_dtype)
    
    return y, last_h

def diagonal_mv(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Optimized diagonal matrix-vector multiplication with dtype handling"""
    assert matrix.device == vector.device, "Inputs must be on same device"
    assert matrix.dtype == vector.dtype, "Inputs must have same dtype"
    assert matrix.shape == vector.shape, "Inputs must have same shape"
    
    supported_dtypes = [torch.float32, torch.float16, torch.bfloat16]
    if matrix.dtype not in supported_dtypes:
        raise ValueError(f"Unsupported dtype {matrix.dtype}. Supported: {supported_dtypes}")
    
    # Convert to float32 if using bfloat16 for stable computation
    orig_dtype = matrix.dtype
    needs_conversion = orig_dtype == torch.bfloat16
    
    if needs_conversion:
        matrix = matrix.float()
        vector = vector.float()
    
    if HAS_CUDA_KERNELS and matrix.is_cuda:
        result = cuda_kernels.diagonal_mv(matrix, vector)
    else:
        result = matrix * vector  # Fallback
    
    if needs_conversion:
        result = result.to(orig_dtype)
    
    return result

def block_diagonal_mm(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Optimized block diagonal matrix multiplication with dtype handling"""
    # Validate inputs
    assert input.device == weight.device, "Inputs must be on same device"
    assert input.dtype == weight.dtype, "Inputs must have same dtype"
    assert weight.dim() == 3, "Weight must be 3D [num_blocks, block_size, block_size]"
    
    supported_dtypes = [torch.float32, torch.float16, torch.bfloat16]
    if input.dtype not in supported_dtypes:
        raise ValueError(f"Unsupported dtype {input.dtype}. Supported: {supported_dtypes}")
    
    orig_dtype = input.dtype
    needs_conversion = orig_dtype == torch.bfloat16
    
    if needs_conversion:
        input = input.float()
        weight = weight.float()
    
    # Reshape input for block processing
    num_blocks = weight.size(0)
    block_size = weight.size(1)
    input_reshaped = input.view(-1, num_blocks, block_size)
    
    if HAS_CUDA_KERNELS and input.is_cuda:
        output = cuda_kernels.block_diagonal_mm(input_reshaped, weight)
    else:
        # Fallback implementation
        output = torch.einsum('...bi,bij->...bj', input_reshaped, weight)
    
    if needs_conversion:
        output = output.to(orig_dtype)
    
    return output.view(*input.shape[:-1], -1)

# ------------------------------
# RMSNorm 
# ------------------------------
class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.width = width
        self.eps = eps
        self.scale = nn.Parameter(torch.empty(width, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and cuda_kernels is not None:
            return cuda_kernels.rms_norm_forward(x, self.scale, self.eps)[0]
        else:
            # Reference implementation exactly:
            x_fp32 = x.float()
            var = torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(var + self.eps)
            scale = self.scale.view(*([1] * (x.ndim - 1) + [-1])).float()
            out_fp32 = x_fp32 * inv_rms * (scale + 1.0)
            # Simulate BF16 rounding
            return out_fp32.to(x.dtype)

# ------------------------------
# BlockDiagonalLinear 
# ------------------------------
class BlockDiagonalLinear(nn.Module):
    """
    Optimized Block Diagonal Linear layer with CUDA support.
    """
    def __init__(self, width: int, num_blocks: int, w_init_variance_scale: float = 1.0,
                 device=None, dtype=None):
        super().__init__()
        self.width = width
        self.num_blocks = num_blocks
        self.w_init_variance_scale = w_init_variance_scale
        self.block_width = self.width // self.num_blocks

        self.w = nn.Parameter(torch.empty([self.num_blocks, self.block_width, self.block_width],
                               device=device, dtype=dtype))
        self.b = nn.Parameter(torch.empty([self.num_blocks, self.block_width],
                               device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = math.sqrt(self.w_init_variance_scale / self.block_width)
        torch.nn.init.normal_(self.w, mean=0.0, std=std)
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if HAS_CUDA_KERNELS and x.is_cuda:
            x_reshaped = x.view(*x.shape[:-1], self.num_blocks, self.block_width)
            y = block_diagonal_mm(x_reshaped, self.w) + self.b
            return y.view(*x.shape[:-1], -1)
        else:
            x_rearr = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)
            y = torch.einsum("... h i, h i j -> ... h j", x_rearr.float(), self.w.float()) + self.b.float()
            y = einops.rearrange(y, "... h j -> ... (h j)")
            return y.to(x.dtype)

# ------------------------------
# RGLRU Module
# ------------------------------
class OptRGLRU(nn.Module):
    def __init__(self, width, num_heads, w_init_variance_scale=1.0, device=None, dtype=None):
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.w_init_variance_scale = w_init_variance_scale

        self.a_param = nn.Parameter(torch.empty(width, device=device, dtype=dtype))
        self.input_gate = BlockDiagonalLinear(width, num_heads, w_init_variance_scale, device, dtype)
        self.a_gate = BlockDiagonalLinear(width, num_heads, w_init_variance_scale, device, dtype)
        self.reset_parameters()

    def reset_parameters(self):
        self.input_gate.reset_parameters()
        self.a_gate.reset_parameters()
        self._a_param_init(self.a_param)

    def _a_param_init(self, w):
        with torch.no_grad():
            w.uniform_(0.9**2 + 1e-8, 0.999**2 + 1e-8)
            w.log_().mul_(0.5)
            w.neg_().exp_().sub_(1.0).log_()

    def forward(self, x: torch.Tensor, segment_pos: torch.Tensor,
                cache: Optional[torch.Tensor] = None,
                return_cache: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bs, seq_len, _ = x.shape
        
        reset = (segment_pos == 0)  # [batch, seq_len] bool tensor
        
        # Compute gates
        gate_x = torch.sigmoid(self.input_gate(x))
        gate_a = torch.sigmoid(self.a_gate(x))

        log_a = -8.0 * gate_a * F.softplus(self.a_param)
        a = torch.exp(log_a)
        a_square = torch.exp(2 * log_a)


        # Gate input and apply normalization
        gated_x = x * gate_x
        multiplier = torch.sqrt(1 - a_square)  
        multiplier = torch.where(reset.unsqueeze(-1), 
                               torch.ones_like(multiplier), 
                               multiplier)
        normalized_x = gated_x * multiplier

        # CUDA-accelerated scan
        y, last_h = rnn_scan(normalized_x, a, reset, cache)

        return (y, last_h) if return_cache else (y, None)

    @staticmethod
    def init_cache(batch_size: int, width: int, device=None):
        return torch.zeros((batch_size, width), dtype=torch.float32, device=device)

# ------------------------------
# Conv1D Module
# ------------------------------
class Conv1D(nn.Module):
    """
    1D Temporal Convolution Layer.

    This implementation follows the reference logic. If running on CUDA,
    you could eventually add a CUDA kernel for Conv1D. For now, it uses the same
    CPU implementation as reference.
    """
    def __init__(self, width: int, temporal_width: int, w_init_variance_scale: float = 0.01,
                 device=None, dtype=None):
        super().__init__()
        self.width = width
        self.temporal_width = temporal_width
        self.w_init_variance_scale = w_init_variance_scale
        
        self.w = nn.Parameter(torch.empty(temporal_width, width, device=device, dtype=dtype))
        self.b = nn.Parameter(torch.empty(width, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(self.w_init_variance_scale / self.temporal_width)
        nn.init.normal_(self.w, mean=0.0, std=std)
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor, segment_pos: torch.Tensor,
                cache: Optional[torch.Tensor] = None,
                return_cache: bool = True) -> Union[Tuple[torch.Tensor, torch.Tensor],
                                                       Tuple[torch.Tensor, None]]:
        batch_size, seq_len, _ = x.shape

        if cache is not None:
            x = torch.cat([cache, x], dim=1)
            prompt_len = self.temporal_width - 1
        else:
            prompt_len = 0
            document_mask = (segment_pos == 0).unsqueeze(-1)
            x = x * (~document_mask).to(x.dtype)

        output = torch.zeros(batch_size, seq_len, self.width, dtype=x.dtype, device=x.device)
        temporal_width = min(self.temporal_width, prompt_len + seq_len)

        for shift in range(temporal_width):
            start = max(prompt_len - shift, 0)
            end = prompt_len + seq_len - shift
            x_window = x[:, start:end]
            L = x_window.size(1)
            if L < seq_len:
                pad_len = seq_len - L
                pad = torch.zeros(batch_size, pad_len, self.width, dtype=x.dtype, device=x.device)
                x_window = torch.cat([pad, x_window], dim=1)
            x_window_fp32 = x_window.float()
            weight_fp32 = self.w[self.temporal_width - shift - 1].float()
            out_chunk = x_window_fp32 * weight_fp32.view(1, 1, -1)
            output += out_chunk.to(x.dtype)

        output += self.b.view(1, 1, -1)

        if not return_cache:
            return output, None
        new_cache = x[:, -self.temporal_width + 1:].clone()

        if new_cache.shape[1] < self.temporal_width - 1:
            pad = torch.zeros(batch_size, self.temporal_width - 1 - new_cache.shape[1],
                              self.width, dtype=new_cache.dtype, device=x.device)
            new_cache = torch.cat([pad, new_cache], dim=1)

        return output, new_cache

    @classmethod
    def init_cache(cls, batch_size: int, width: int, dtype: torch.dtype,
                   conv1d_temporal_width: int = 4, device=None) -> torch.Tensor:
        return torch.zeros((batch_size, conv1d_temporal_width - 1, width), dtype=dtype, device=device)

# ------------------------------
# LocalAttentionBlock Module
# ------------------------------
class OptLocalAttentionBlock(nn.Module):
    """
    Local Multi-Head Attention block.

    This module implements the local attention operator. When running on CUDA
    and if a single token is processed (in decoding mode), the optimized attention
    kernel from attention_kernels is used.
    """
    def __init__(self, width: int, num_heads: int, window_size: int,
                 final_w_init_variance_scale: float = 1.0,
                 device=None, dtype=None):
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.window_size = window_size
        self.final_w_init_variance_scale = final_w_init_variance_scale

        self.proj_q = nn.Linear(width, width, bias=False, device=device, dtype=dtype)
        self.proj_k = nn.Linear(width, self.head_dim, bias=False, device=device, dtype=dtype)
        self.proj_v = nn.Linear(width, self.head_dim, bias=False, device=device, dtype=dtype)

        self.proj_final = nn.Linear(width, width, bias=True, device=device, dtype=dtype)

    def reset_parameters(self):
        nn.init.normal_(self.proj_q.weight, std=math.sqrt(1.0 / self.width))
        nn.init.normal_(self.proj_k.weight, std=math.sqrt(1.0 / self.width))
        nn.init.normal_(self.proj_v.weight, std=math.sqrt(1.0 / self.width))
        std = math.sqrt(self.final_w_init_variance_scale / self.width)
        nn.init.normal_(self.proj_final.weight, std=std)
        nn.init.zeros_(self.proj_final.bias)

    @classmethod
    def init_cache(cls, batch_size: int, window_size: int, heads_dim: int, dtype: torch.dtype, device=None) -> dict:
        return {
            "keys": torch.zeros((batch_size, window_size, 1, heads_dim), dtype=dtype, device=device),
            "values": torch.zeros((batch_size, window_size, 1, heads_dim), dtype=dtype, device=device),
            "num_tokens": torch.zeros((batch_size,), dtype=torch.int32, device=device),
        }
    
    def _apply_rope(self, inputs: torch.Tensor, positions: torch.Tensor, max_wavelength: int = 10000) -> torch.Tensor:
        """
        Applies RoPE as in the reference implementation.
        Expects inputs of shape [batch, seq_len, num_heads, head_dim].
        Splits the inputs into a rope portion and a pass-through portion,
        applies the rotary embeddings on the rope portion, and then
        concatenates the result with the pass-through part.
        """
        batch_size, seq_len, num_heads, head_dim = inputs.shape
        # Split inputs into the portion that will have RoPE applied and the remainder.
        # In the reference, the input is split into two halves along the last dimension.
        x_rope, x_pass = torch.chunk(inputs, 2, dim=-1)  # x_rope: [b, t, n, head_dim/2]
        
        # Reshape positions to broadcast over the head dimensions.
        positions = positions.view(batch_size, seq_len, 1, 1)
        
        # Determine frequencies
        # The reference uses half the dimension of x_rope for frequencies.
        freq = torch.arange(x_rope.shape[-1] // 2, device=inputs.device, dtype=torch.float32)
        freq_exponents = 2 * freq / x_rope.shape[-1]  # Note: x_rope.shape[-1] == head_dim/2
        timescale = max_wavelength ** freq_exponents
        inv_frequencies = 1.0 / timescale  # shape: [head_dim/4]
        
        # Compute sinusoid terms.
        sinusoid_imp = positions * inv_frequencies.view(1, 1, 1, -1)
        sin = torch.sin(sinusoid_imp).type_as(inputs)
        cos = torch.cos(sinusoid_imp).type_as(inputs)
        
        first_half, second_half = torch.chunk(x_rope, 2, dim=-1)  # each of shape [b, t, n, head_dim/4]
        
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        
        # Concatenate the two transformed parts and then the pass-through.
        return torch.cat([first_part, second_part, x_pass], dim=-1)


    def _compute_mask(self, segment_pos: torch.Tensor, cache: Optional[dict] = None) -> torch.Tensor:
        if cache is None:
            positions = torch.arange(segment_pos.size(1), device=segment_pos.device)
            positions = positions.unsqueeze(0).expand_as(segment_pos)
            segment_ids = torch.cumsum(segment_pos == 0, dim=-1)

            causal = positions.unsqueeze(-1) >= positions.unsqueeze(-2)
            window = positions.unsqueeze(-1) <= (positions.unsqueeze(-2) + self.window_size)
            segment = segment_ids.unsqueeze(-1) == segment_ids.unsqueeze(-2)
            mask = causal & window & segment
            
            return mask.unsqueeze(1)
        else:
            return torch.ones((segment_pos.size(0), 1, 1, self.window_size),
                              dtype=torch.bool, device=segment_pos.device)

    def forward(self, x: torch.Tensor, segment_pos: torch.Tensor,
                cache: Optional[dict] = None, return_cache: bool = True) -> Union[Tuple[torch.Tensor, dict],
                                                                                  Tuple[torch.Tensor, None]]:
        batch_size, seq_len, _ = x.shape
        # Query projection: shape [batch, seq_len, num_heads, head_dim]
        q = self.proj_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Key and value projections: first produce shape [batch, seq_len, 1, head_dim],
        # then expand to [batch, seq_len, num_heads, head_dim]
        k = self.proj_k(x).view(batch_size, seq_len, 1, self.head_dim).expand(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.proj_v(x).view(batch_size, seq_len, 1, self.head_dim).expand(batch_size, seq_len, self.num_heads, self.head_dim)
        q = self._apply_rope(q, segment_pos)
        k = self._apply_rope(k, segment_pos)

        if cache is not None:
            k = torch.cat([cache["keys"], k], dim=1)
            v = torch.cat([cache["values"], v], dim=1)
            new_cache = {
                "keys": k[:, -self.window_size:],
                "values": v[:, -self.window_size:],
                "num_tokens": cache["num_tokens"] + seq_len
            }
        else:
            new_cache = None

        if HAS_ATTENTION_KERNELS and x.is_cuda and seq_len == 1:
            scale = 1.0 / math.sqrt(self.head_dim)
            # Rearrange q, k, v from (batch, 1, num_heads, head_dim) to (batch * num_heads, head_dim)
            q_reshaped = q.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, self.head_dim)
            k_reshaped = k.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, -1, self.head_dim)
            v_reshaped = v.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, -1, self.head_dim)
            output, _ = attention_kernels.attention_forward(
                q_reshaped, k_reshaped, v_reshaped, scale
            )
            output = output.to(self.proj_final.weight.dtype)
            # Reshape output: from (batch*num_heads, head_dim) to (batch, num_heads, head_dim),
            # then flatten the heads dimension to get shape (batch, 1, width)
            attn_output = output.view(batch_size, self.num_heads, self.head_dim).reshape(batch_size, 1, self.width)
        else:
            logits = torch.einsum("bqnh,bknh->bnqk", q, k) * (self.head_dim ** -0.5)
            mask = self._compute_mask(segment_pos, cache)
            logits = torch.where(mask, logits, torch.finfo(logits.dtype).min)
            probs = F.softmax(logits.float(), dim=-1).to(x.dtype)
            attn_output = torch.einsum("bnqk,bknh->bqnh", probs, v)
            attn_output = attn_output.reshape(batch_size, seq_len, self.width)

        output = self.proj_final(attn_output)
        return (output, new_cache) if return_cache else (output, None)

# ------------------------------
# ResidualBlock Module
# ------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, width: int, mlp_expanded_width: int, temporal_block_type: str,
                 attention_window_size: int, recurrent_num_heads: int = 10,
                 attention_num_heads: int = 40, lru_width: Optional[int] = None,
                 conv1d_temporal_width: int = 4, final_w_init_variance_scale: float = 1.0,
                 device=None, dtype=None):
        super().__init__()
        self.width = width
        self.mlp_expanded_width = mlp_expanded_width
        self.temporal_block_type = temporal_block_type
        self.attention_window_size = attention_window_size
        self.recurrent_num_heads = recurrent_num_heads
        self.attention_num_heads = attention_num_heads
        self.lru_width = lru_width
        self.conv1d_temporal_width = conv1d_temporal_width
        self.final_w_init_variance_scale = final_w_init_variance_scale

        self.temporal_pre_norm = RMSNorm(width, device=device, dtype=dtype)
        if temporal_block_type == "recurrent":
            self.temporal_block = OptRGLRU(
                width=width, num_heads=recurrent_num_heads, 
                w_init_variance_scale=final_w_init_variance_scale,
                device=device, dtype=dtype
            )
        elif temporal_block_type == "attention":
            self.temporal_block = OptLocalAttentionBlock(
                width=width, num_heads=attention_num_heads,
                window_size=attention_window_size, 
                final_w_init_variance_scale=final_w_init_variance_scale,
                device=device, dtype=dtype
            )
        else:
            raise ValueError(f"Unknown block type: {temporal_block_type}")
        self.channel_pre_norm = RMSNorm(width, device=device, dtype=dtype)
        self.mlp_block = MLPBlock(
            width=width, expanded_width=mlp_expanded_width,
            final_w_init_variance_scale=final_w_init_variance_scale,
            device=device, dtype=dtype
        )

    def reset_parameters(self):
        self.temporal_pre_norm.reset_parameters()
        self.temporal_block.reset_parameters()
        self.channel_pre_norm.reset_parameters()
        self.mlp_block.reset_parameters()

    def forward(self, x: torch.Tensor, segment_pos: torch.Tensor,
                cache: Optional[Dict[str, torch.Tensor]] = None,
                return_cache: bool = True) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                                                       Tuple[torch.Tensor, None]]:
        residual = x
        x = self.temporal_pre_norm(x)
        x, cache = self.temporal_block(x, segment_pos, cache, return_cache)
        x = x + residual

        residual = x
        x = self.channel_pre_norm(x)
        x = self.mlp_block(x)
        x = x + residual

        return x, cache

    @classmethod
    def init_cache(cls, batch_size: int, width: int, num_heads: int, attention_window_size: int,
                   temporal_block_type: str, dtype: torch.dtype, lru_width: Optional[int] = None,
                   conv1d_temporal_width: int = 4, device=None) -> Dict[str, torch.Tensor]:
        if temporal_block_type == "recurrent":
            return OptRGLRU.init_cache(batch_size, width, device)
        elif temporal_block_type == "attention":
            return OptLocalAttentionBlock.init_cache(batch_size, attention_window_size, width // num_heads, dtype, device)
        else:
            raise ValueError(f"Unknown block type: {temporal_block_type}")

# ----------------------------
# OptimizedRecurrentGemma
# ------------------------------
class OptimizedRecurrentGemma(nn.Module):
    def __init__(self, config, gradient_checkpointing: bool = False, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing
        self.embedder = Embedder(
            vocab_size=config.vocab_size,
            embed_dim=config.width,
            scale_by_sqrt_dim=config.embeddings_scale_by_sqrt_dim,
            device=device,
            dtype=dtype,
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(
                width=config.width,
                mlp_expanded_width=config.mlp_expanded_width,
                temporal_block_type=block_type,
                attention_window_size=config.attention_window_size,
                recurrent_num_heads=config.recurrent_num_heads,
                attention_num_heads=config.attention_num_heads,
                lru_width=config.lru_width,
                conv1d_temporal_width=config.conv1d_temporal_width,
                final_w_init_variance_scale=2.0 / config.num_layers,
                device=device,
                dtype=dtype,
            )
            for block_type in config.block_types
        ])
        self.final_norm = RMSNorm(config.width, device=device, dtype=dtype)

    def reset_parameters(self):
        self.embedder.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()
        self.final_norm.reset_parameters()

    def init_cache(self, batch_size: int, dtype: torch.dtype) -> Dict[str, Dict[str, torch.Tensor]]:
        cache = {}
        for i, block_type in enumerate(self.config.block_types):
            num_heads = (self.config.recurrent_num_heads if block_type == "recurrent"
                         else self.config.attention_num_heads)
            cache[f"blocks.{i}"] = ResidualBlock.init_cache(
                batch_size=batch_size,
                width=self.config.width,
                num_heads=num_heads,
                attention_window_size=self.config.attention_window_size,
                temporal_block_type=block_type,
                dtype=dtype,
                lru_width=self.config.lru_width,
                conv1d_temporal_width=self.config.conv1d_temporal_width,
                device=self.embedder.input_embedding.device,
            )
        return cache

    def forward(self, tokens: torch.Tensor, segment_pos: torch.Tensor,
                cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
                return_logits: bool = True, return_cache: bool = True) -> Union[
                    Tuple[None, None],
                    Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]]:
        x = self.embedder.encode(tokens)
        new_cache = {}
        for i, block in enumerate(self.blocks):
            block_name = f"blocks.{i}"
            block_cache = None if cache is None else cache[block_name]
            if self.gradient_checkpointing and self.training:
                x, new_cache[block_name] = torch.utils.checkpoint.checkpoint(
                    block, x, segment_pos, block_cache, return_cache, use_reentrant=False)
            else:
                x, new_cache[block_name] = block(x, segment_pos, block_cache, return_cache)
        x = self.final_norm(x)
        logits = self.embedder.decode(x)
        if self.config.logits_soft_cap > 0:
            logits = torch.tanh(logits / self.config.logits_soft_cap) * self.config.logits_soft_cap
        return logits, new_cache

    def generate(self, tokens: torch.Tensor, max_new_tokens: int = 1024) -> torch.Tensor:
        """
        Autoregressive generation with greedy sampling.
        
        Given an initial prompt (tokens) with shape (batch, seq_len), this method
        autoregressively generates max_new_tokens tokens by taking the argmax of the logits
        at each step and appending it to the sequence.
        
        Returns:
        A tensor containing the original tokens and the generated tokens.
        """
        self.eval()
        with torch.no_grad():
            # Initialize cache using the provided dtype
            cache = self.init_cache(tokens.size(0), self.embedder.input_embedding.dtype)
            output = tokens
            # Create segment positions for the prompt
            prompt_seg = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0).expand(tokens.size(0), -1)
            # Process the prompt to update the cache.
            _, cache = self(tokens, prompt_seg, cache=None, return_logits=False, return_cache=True)
            for _ in range(max_new_tokens):
                last_token = output[:, -1:]  # shape: (batch, 1)
                new_seg = torch.full((tokens.size(0), 1), output.size(1), device=tokens.device, dtype=tokens.dtype)
                logits, cache = self(last_token, new_seg, cache=cache, return_cache=True, return_logits=True)
                next_token = torch.argmax(logits, dim=-1)  # shape: (batch, 1) if logits was (batch, 1, vocab_size)
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(1)
                output = torch.cat([output, next_token], dim=1)
            return output
