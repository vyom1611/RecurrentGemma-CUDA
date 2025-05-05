# RecurrentGemma CUDA-Accelerated Optimization

Highly optimized PyTorch + CUDA implementation of the RecurrentGemma (Griffin) language model, leveraging advanced GPU features for maximum inference throughput.
Built upon the original RecurrentGemma repository found at https://github.com/google-deepmind/recurrentgemma.
(Note: modules.py and layers.py are taken from original/reference implementation with no edits)

---

## Key CUDA Optimizations

| Operator                | GPU Feature            | Optimization Strategy                                         |
|-------------------------|------------------------|---------------------------------------------------------------|
| **Block-Diagonal MM**   | Coalesced Loads        | 1D kernel: one thread per output element, contiguous memory reads. Loop unrolling over small fixed block_size.  |
| **Diagonal MV**         | Elementwise Parallel   | Simple kernel: each thread scales one element, avoids zero multiplications.                                     |
| **RNN Scan**            | Thread-Level Parallelism | 2D grid: threads = B×F, each thread does serial T-step recurrence in registers, no atomics needed.           |
| **RMSNorm**             | Shared Memory Reduction | Block per token: lanes compute partial sums, tree-reduce in shared memory, FP32 accumulation, then elementwise norm. |
| **Attention (Dot Prod)**| WMMA / Tensor Cores     | Tile size 16×16: load_matrix_sync for tiles, mma_sync for FP32 accumulate, half→half store for scores.        |
| **Softmax**             | Warp & Shared Memory    | Per-row block: warp parallel max+sum reduction in shared memory, then normalize in-place.                    |
| **Attention (Weighted Sum)**| WMMA / Tensor Cores | Tile-based A×V: load tiles of softmax_scores & V into fragments, mma_sync accumulation, coalesced store.     |

---

## Design Patterns & GPU Considerations

- **Coalesced Memory Access**: Data layouts ensure adjacent threads access consecutive addresses (e.g., flat `[B*T*F]` indexing or tile strides matching memory).
- **Shared Memory**: Used in RMSNorm and Softmax for intra-block reductions to minimize global memory traffic and latency.
- **Tensor Cores (WMMA API)**: Leverage half-precision (FP16/ BF16) matrix‐multiply‐accumulate with FP32 accumulation for attention dot‐products and weighted sums.
- **Register-Resident State**: In `rnn_scan`, hidden states kept in registers within each thread, enabling fast serial recurrence without inter-thread synchronization.
- **No Atomics**: Independence of features avoids atomic operations, preserving full parallel throughput.
- **Fixed-Size Tiling**: Knowledge of window sizes (e.g., attention_window=2048, WMMA=16) enables loop bounds to be constant and kernels to be tuned.

---

## Inference Optimizations

- **KV Caching**:  Store and reuse keys/values per attention head in fast GPU memory. On each new token, only compute its key/value and append, then maintain a fixed-size sliding window (window_size=2048) to avoid recomputing the entire context.
- **Hidden-State Caching**:  For RGLRU, keep the last hidden state (`h_t`) in registers per [batch,feature] thread. Subsequent recurrences start from this cached state without re-scanning the prompt.
- **Greedy Decoding with Minimal Overhead**:  Use `argmax` on logits to select next token, minimizing branching and control flow divergence in kernels.
- **Batch Size vs. Sequence Length**: Tune grid/block dimensions such that batch and feature parallelism saturates SMs even for small generation batches.

---

## Usage Example

```python
import torch
from optimized_recurrentgemma.optimized_recurrentgemma import GriffinConfig, OptimizedRecurrentGemma

# Configure and build model
config = GriffinConfig.from_preset(vocab_size=256000, preset="RECURRENT_GEMMA_2B_V1")
model = OptimizedRecurrentGemma(config).cuda().eval()

# Forward pass
tokens = torch.randint(0, config.vocab_size, (2, 512), device="cuda")
seg_pos = torch.arange(512, device="cuda").unsqueeze(0).expand(2, -1)
logits, cache = model(tokens, seg_pos)

# Generative inference
generated = model.generate(tokens, max_new_tokens=128)
```

---

## Benchmark & Testing

- **Throughput Measurement**: Tokens/sec for (B=2, context=4096, gen=1024).
    - Speedups:
        - Total Throughput speedup: 1.71
        - Without using custom CUDA kernels and just torch.compile: 1.28
        
- **Correctness**: Kernel outputs validated with `torch.allclose` (rtol=3e-3, atol=5e-3). Top-5 optimized logits in Top-10 reference logits.
- **Run:**
  ```bash
  pytest test_monolith.py
  ```

---

## License

Apache 2.0
