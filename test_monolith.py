#!/usr/bin/env python3
import torch
import time
import math
from typing import Tuple
from torch import nn

from recurrentgemma.torch.layers import (
    rnn_scan,  
    RMSNorm as OriginalRMSNorm,
    BlockDiagonalLinear as RefBlockDiagonalLinear
)
from recurrentgemma.torch.modules import LocalAttentionBlock as OrigLocalAttentionBlock

from recurrentgemma import torch as recurrentgemma

from optimized_recurrentgemma.optimized_recurrentgemma import (
    OptimizedRecurrentGemma,
    GriffinConfig,
    OptLocalAttentionBlock as OptLocalAttentionBlock,
    BlockDiagonalLinear as OptBlockDiagonalLinear,
    rnn_scan as rnn_scan_opt
)

try:
    from optimized_recurrentgemma.optimized_recurrentgemma import cuda_kernels
    HAS_CUDA_KERNELS = True
except ImportError:
    print("WARNING: CUDA kernels not available!")
    HAS_CUDA_KERNELS = False


def setup_models(device: str = "cuda", dtype=torch.bfloat16):
    """Initialize both the reference and optimized models with identical configuration."""
    vocab_size = 256000

    ref_model = None
    config = recurrentgemma.GriffinConfig.from_preset(
            vocab_size=vocab_size,
            preset=recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1,
        )
    ref_model = recurrentgemma.Griffin(config).to(device=device, dtype=dtype).eval()
    if hasattr(ref_model, "gradient_checkpointing"):
            ref_model.gradient_checkpointing = False

    opt_config = GriffinConfig.from_preset(vocab_size=vocab_size, preset="RECURRENT_GEMMA_2B_V1")
    opt_model = OptimizedRecurrentGemma(opt_config).to(device=device, dtype=dtype).eval()

    return ref_model, opt_model

# Test 1: Top-k Logits Consistency
def test_topk_logits(ref_model, opt_model, batch_size=2, seq_len=64):
    device = next(opt_model.parameters()).device
    tokens = torch.randint(0, opt_model.config.vocab_size, (batch_size, seq_len), device=device)
    segment_pos = torch.arange(seq_len, device=device).expand(batch_size, -1)

    with torch.no_grad():
        ref_logits, _ = ref_model(tokens, segment_pos=segment_pos)
        opt_logits, _ = opt_model(tokens, segment_pos)

    print("\n=== Strict Top-k Logits Test ===")
    print("1. First check if top-5 opt matches top-5 ref exactly")
    print("2. If not, check if top-5 opt is within top-10 ref")
    
    # Get top indices
    ref_top5 = torch.topk(ref_logits, 5, dim=-1).indices
    ref_top10 = torch.topk(ref_logits, 10, dim=-1).indices
    opt_top5 = torch.topk(opt_logits, 5, dim=-1).indices

    # Check 1: Exact top-5 match
    exact_matches = torch.all(opt_top5 == ref_top5, dim=-1)
    exact_match_rate = exact_matches.float().mean().item() * 100
    print(f"\nExact top-5 match rate: {exact_match_rate:.1f}%")

    # Check 2: Top-5 within top-10 (where not exact match)
    fallback_check = torch.isin(opt_top5, ref_top10)
    fallback_match_rate = fallback_check.float().mean().item() * 100
    print(f"Fallback top-5-in-top-10 match rate: {fallback_match_rate:.1f}%")

    # Find and report mismatches
    mismatches = []
    for b in range(batch_size):
        for p in range(seq_len):
            if not exact_matches[b,p]:
                if not torch.all(torch.isin(opt_top5[b,p], ref_top10[b,p])):
                    mismatches.append((b, p))
                    if len(mismatches) <= 3:
                        print(f"\nMismatch at batch {b}, position {p}:")
                        print(f"Reference top-5: {ref_top5[b,p].tolist()}")
                        print(f"Reference top-10: {ref_top10[b,p].tolist()}")
                        print(f"Optimized top-5: {opt_top5[b,p].tolist()}")
                        print("Missing tokens:", 
                              [t for t in opt_top5[b,p] if t not in ref_top10[b,p]])

    # Final verdict
    if len(mismatches) == 0:
        print("\n✅ All optimized top-5 tokens:")
        print("   - Either exactly match reference top-5")
        print("   - Or are within reference top-10")
    else:
        print(f"\n❌ Found {len(mismatches)} positions where:")
        print("   - Optimized top-5 doesn't match reference top-5")
        print("   - AND contains tokens outside reference top-10")

    # Numerical differences for debugging
    diff = (ref_logits - opt_logits).abs()
    print(f"\nNumerical differences (max/mean): {diff.max().item():.4f}/{diff.mean().item():.4f}")


# Test 2: Compare Temporal RNN Scan Kernels (GPU)
def test_temporal_rnn_scan():
    if not HAS_CUDA_KERNELS:
        print("Skipping temporal_rnn_scan test; CUDA kernels not available.")
        return

    print("\n=== Testing temporal_rnn_scan (GPU) ===")
    batch, t, d = 2, 10, 16
    
    # Create BF16 inputs on GPU
    input_tensor = torch.randn(batch, t, d, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(batch, t, d, device="cuda", dtype=torch.bfloat16)
    reset = torch.zeros(batch, t, device="cuda", dtype=torch.bool)  
    
    # The original repo's rnn_scan expects h0 as float32
    init_state = torch.zeros(batch, d, device="cuda", dtype=torch.float32)
    
    init_state_bf16 = init_state.to(torch.bfloat16)

    out_cuda, final_state_cuda = rnn_scan_opt(input_tensor, a, reset, init_state_bf16)
    out_cuda = out_cuda.float()
    final_state_cuda = final_state_cuda.float()

    out_ref, final_state_ref = rnn_scan(input_tensor, a, reset=reset, h0=init_state)
    out_ref = out_ref.float()
    final_state_ref = final_state_ref.float()

    if torch.allclose(out_cuda, out_ref, rtol=3e-3, atol=5e-3) and torch.allclose(final_state_cuda, final_state_ref, rtol=3e-3, atol=5e-3):
        print("temporal_rnn_scan outputs match the original implementation within tolerance.")
    else:
        print("temporal_rnn_scan outputs differ!")
        print("Max diff (output):", torch.max(torch.abs(out_cuda - out_ref)).item())
        print("Max diff (final state):", torch.max(torch.abs(final_state_cuda - final_state_ref)).item())

# Test 3: Compare RMSNorm Kernels (GPU)
def test_rms_norm_forward():
    if not HAS_CUDA_KERNELS:
        print("Skipping rms_norm_forward test; CUDA kernels not available.")
        return

    print("\n=== Testing rms_norm_forward (GPU) ===")
    N, d = 32, 64
    input_tensor = torch.randn(N, d, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(d, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6
    out_cuda = cuda_kernels.rms_norm_forward(input_tensor, scale, eps)[0].float()

    orig_rms_norm = OriginalRMSNorm(width=d, eps=eps, device="cuda", dtype=torch.bfloat16).to("cuda")
    with torch.no_grad():
        orig_rms_norm.scale.copy_(scale)
    out_ref = orig_rms_norm(input_tensor).float()

    if torch.allclose(out_cuda, out_ref, rtol=3e-2, atol=5e-2):
        print("rms_norm_forward outputs match the original implementation within tolerance.")
    else:
        print("rms_norm_forward outputs differ!")
        print("Max diff:", torch.max(torch.abs(out_cuda - out_ref)).item())

def test_attention_block():
    print("\n=== Testing Attention Block ===")
    batch_size = 2
    seq_len = 64
    width = 2560
    num_heads = 10
    window_size = 2048
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, width, device=device, dtype=dtype)
    segment_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    orig_attn_block = OrigLocalAttentionBlock(
        width=width,
        num_heads=num_heads,
        window_size=window_size,
        device=device,
        dtype=dtype,
    ).eval()

    opt_attn_block = OptLocalAttentionBlock(
        width=width,
        num_heads=num_heads,
        window_size=window_size,
        device=device,
        dtype=dtype,
    ).eval()

    def get_weights(module):
        """Extract weights in consistent order"""
        return {
            'q_weight': module.proj_q.weight.data.clone(),
            'k_weight': module.proj_k.weight.data.clone(),
            'v_weight': module.proj_v.weight.data.clone(),
            'final_weight': module.proj_final.weight.data.clone(),
            'final_bias': module.proj_final.bias.data.clone()
        }

    # Get original weights
    orig_weights = get_weights(orig_attn_block)

    # Apply to optimized block
    with torch.no_grad():
        opt_attn_block.proj_q.weight.copy_(orig_weights['q_weight'])
        opt_attn_block.proj_k.weight.copy_(orig_weights['k_weight'])
        opt_attn_block.proj_v.weight.copy_(orig_weights['v_weight'])
        opt_attn_block.proj_final.weight.copy_(orig_weights['final_weight'])
        opt_attn_block.proj_final.bias.copy_(orig_weights['final_bias'])

    # Verify weight copying
    opt_weights = get_weights(opt_attn_block)
    for key in orig_weights:
        assert torch.allclose(orig_weights[key], opt_weights[key]), f"Weight mismatch in {key}"

    with torch.no_grad():
        orig_out, _ = orig_attn_block(x, segment_pos)
        opt_out, _ = opt_attn_block(x, segment_pos)

    diff = (orig_out - opt_out).abs()

    atol = 5e-3  
    rtol = 3e-3  

    close = torch.allclose(orig_out, opt_out, rtol=rtol, atol=atol)
    if close:
        print("Outputs match within tolerance")
    else:
        print("Significant differences detected")
        top_diffs, indices = torch.topk(diff.view(-1), 5)
        for i in range(5):
            idx = indices[i]
            print(f"Diff {i+1}: {top_diffs[i].item():.6f}")
            print(f"Original: {orig_out.view(-1)[idx].item():.6f}")
            print(f"Optimized: {opt_out.view(-1)[idx].item():.6f}")

    return close

def test_block_diagonal_linear():
    print("\n=== Testing BlockDiagonalLinear Module ===")
    width = 32
    num_blocks = 4
    batch, seq_len = 2, 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    x = torch.randn(batch, seq_len, width, device=device, dtype=dtype)

    # Initialize modules with same weights
    torch.manual_seed(42)
    ref_linear = RefBlockDiagonalLinear(width, num_blocks).to(device=device, dtype=dtype)
    
    torch.manual_seed(42)
    opt_linear = OptBlockDiagonalLinear(width, num_blocks).to(device=device, dtype=dtype)

    ref_output = ref_linear(x)

    opt_output = opt_linear(x)

    if torch.allclose(opt_output, ref_output, rtol=1e-2, atol=1e-2):
        print("BlockDiagonalLinear outputs match reference implementation.")
    else:
        print("BlockDiagonalLinear outputs differ!")
        print("Max diff:", torch.max(torch.abs(opt_output - ref_output)).item())


# Test 4: Throughput Measurement
def measure_throughput(model, batch_size=2, context_length=4096, gen_length=1024):
    """Measure generation throughput"""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    tokens = torch.randint(
        low=0, high=model.config.vocab_size,
        size=(batch_size, context_length),
        device=device
    )
    segment_pos = torch.arange(context_length, device=device).expand(batch_size, -1)

    with torch.no_grad():
        if hasattr(model, 'init_cache'):
            _, cache = model(tokens, segment_pos=segment_pos)
        else:
            _, cache = model(tokens, segment_pos=segment_pos)

    gen_input = tokens[:, -1:]
    segment_pos = segment_pos[:, -1:] + 1

    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        for _ in range(gen_length):
            if hasattr(model, 'init_cache'):
                logits, cache = model(gen_input, segment_pos=segment_pos, cache=cache)
            else:
                logits, cache = model(gen_input, segment_pos=segment_pos, cache=cache)
            
            gen_input = torch.argmax(logits[:, -1:], dim=-1)
            segment_pos = segment_pos + 1
    
    torch.cuda.synchronize()
    end = time.time()

    return (batch_size * gen_length) / (end - start)



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=== Setting up models on GPU ===")
    ref_model, opt_model = setup_models(device, dtype)

    print("\n=== Test 1: Top-k Logits Consistency ===")
    test_topk_logits(ref_model, opt_model)

    print("\n=== Test 2: temporal_rnn_scan Kernel Comparison ===")
    test_temporal_rnn_scan()

    print("\n=== Test 3: rms_norm_forward Kernel Comparison ===")
    test_rms_norm_forward()

    print("\n=== Test 4: Attention Kernel Comparison ===")
    test_attention_block()

    print("\n=== Test 5: Block Diagonal Kernel Comparison ===")
    test_block_diagonal_linear()

    print("\n=== Test 6: Throughput Measurement ===")
    batch_size = 2
    context_length = 4096
    gen_length = 1024

    print("Measuring reference implementation throughput...")
    ref_throughput = measure_throughput(ref_model, batch_size, context_length, gen_length)
    print(f"Reference throughput: {ref_throughput:.2f} tokens/sec")

    print("Measuring optimized implementation throughput...")
    opt_throughput = measure_throughput(opt_model, batch_size, context_length, gen_length)
    print(f"Optimized throughput: {opt_throughput:.2f} tokens/sec")

    if ref_throughput:
        speedup = opt_throughput / ref_throughput
        print(f"Speedup: {speedup:.2f}x")
    else:
        print("Reference throughput not available; cannot compute speedup.")

    print("\nAll tests completed!")
