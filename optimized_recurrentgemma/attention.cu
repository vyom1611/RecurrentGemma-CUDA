#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <cmath>

using namespace nvcuda::wmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ---------------------------------------------------------------------
// Kernel 1: Compute Q*K^T (scaled) using tensor cores with FP32 accumulation.
// queries: [num_rows, head_dim] in row-major (__half)
// keys: [num_cols, head_dim] in row-major (__half) -- loaded as column-major
// attn_scores: [num_rows, num_cols] in row-major (__half)
// scale: scaling factor (1/sqrt(head_dim))
template <typename scalar_t>
__global__ void attention_dot_product_kernel_fp32(
    const scalar_t* __restrict__ queries,
    const scalar_t* __restrict__ keys,
    scalar_t* __restrict__ attn_scores,
    int num_rows, int num_cols, int head_dim,
    float scale
) {
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;
    
    int row_start = tile_row * WMMA_M;
    int col_start = tile_col * WMMA_N;
  
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    fill_fragment(c_frag, 0.0f);
    
    // Loop over the head dimension in tiles of WMMA_K
    for (int k_tile = 0; k_tile < head_dim; k_tile += WMMA_K) {
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
        // For keys, load as column-major so that we effectively get K^T
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
        
        const __half* query_tile = queries + row_start * head_dim + k_tile;
        const __half* key_tile = keys + col_start * head_dim + k_tile;
        
        load_matrix_sync(a_frag, query_tile, head_dim);
        load_matrix_sync(b_frag, key_tile, head_dim);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Scale the accumulated result in FP32
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] = c_frag.x[i] * scale;
    }
    
    // Now, convert the FP32 result to __half
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, __half> out_frag;
    for (int i = 0; i < c_frag.num_elements; i++) {
        out_frag.x[i] = __float2half(c_frag.x[i]);
    }
    
    __half* out_tile = attn_scores + row_start * num_cols + col_start;
    store_matrix_sync(out_tile, out_frag, num_cols, mem_row_major);
}

// ---------------------------------------------------------------------
// Kernel 2: Softmax per row
// This kernel is launched with one block per row.
// It uses shared memory to compute the row-wise max and sum.
__global__ void softmax_kernel(
    __half* scores,  // [num_rows, num_cols]
    int num_rows,
    int num_cols
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;
    
    extern __shared__ float shared_mem[];  
    __half* row_scores = scores + row * num_cols;
    
    float max_val = -1e20f;
    for (int j = threadIdx.x; j < num_cols; j += blockDim.x) {
        float val = __half2float(row_scores[j]);
        if (val > max_val) max_val = val;
    }
    shared_mem[threadIdx.x] = max_val;
    __syncthreads();
    
    // Parallel reduction to find max
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride)
            shared_mem[threadIdx.x] = fmaxf(shared_mem[threadIdx.x], shared_mem[threadIdx.x + stride]);
        __syncthreads();
    }
    max_val = shared_mem[0];
    
    // Compute exponentials and partial sum
    float sum = 0.0f;
    for (int j = threadIdx.x; j < num_cols; j += blockDim.x) {
        float val = __half2float(row_scores[j]);
        float exp_val = expf(val - max_val);
        row_scores[j] = __float2half(exp_val);
        sum += exp_val;
    }
    shared_mem[threadIdx.x] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride)
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
        __syncthreads();
    }
    sum = shared_mem[0];
    
    // Normalize row
    for (int j = threadIdx.x; j < num_cols; j += blockDim.x) {
        float exp_val = __half2float(row_scores[j]);
        row_scores[j] = __float2half(exp_val / sum);
    }
}

// ---------------------------------------------------------------------
// Kernel 3: Weighted Sum: Compute output = softmax_scores * V using WMMA.
// softmax_scores: [num_rows, num_cols] in row-major
// V: [num_cols, head_dim] in row-major
// output: [num_rows, head_dim] in row-major
__global__ void attention_weighted_sum_kernel(
    const __half* __restrict__ softmax_scores,  // [num_rows, num_cols]
    const __half* __restrict__ V,               // [num_cols, head_dim]
    __half* __restrict__ output,                // [num_rows, head_dim]
    int num_rows, int num_cols, int head_dim
) {
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;
    
    int row_start = tile_row * WMMA_M;
    int col_start = tile_col * WMMA_N;
    
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag;
    fill_fragment(c_frag, __float2half(0.0f));

    for (int k_tile = 0; k_tile < num_cols; k_tile += WMMA_K) {
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
        
        const __half* a_tile = softmax_scores + row_start * num_cols + k_tile;
        const __half* b_tile = V + col_start * num_cols + k_tile;
        
        load_matrix_sync(a_frag, a_tile, num_cols);
        load_matrix_sync(b_frag, b_tile, num_cols);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    __half* out_tile = output + row_start * head_dim + col_start;
    store_matrix_sync(out_tile, c_frag, head_dim, mem_row_major);
}

// ---------------------------------------------------------------------
// Host function: Full Attention Forward Pass using tensor cores.
// Expects queries, keys, and V as BF16 (i.e. __half) tensors.
// Returns a vector of two tensors: [output, attention_scores].
// - output: [num_rows, head_dim]
// - attention_scores: [num_rows, num_cols] (after softmax)
std::vector<torch::Tensor> attention_forward(
    torch::Tensor queries,  
    torch::Tensor keys,     
    torch::Tensor V,        
    float scale             
) {
    int num_rows = queries.size(0);
    int head_dim = queries.size(1);
    int num_cols = keys.size(0);

    // Create output tensor for attention scores
    auto attn_scores = torch::empty({num_rows, num_cols}, queries.options());

    int threads = 256;
    int blocks_x = (num_cols + WMMA_N - 1) / WMMA_N;
    int blocks_y = (num_rows + WMMA_M - 1) / WMMA_M;
    dim3 blocks(blocks_x, blocks_y);

    if (queries.scalar_type() != torch::kHalf) {
      queries = queries.to(torch::kHalf);
    }
    if (keys.scalar_type() != torch::kHalf) {
      keys = keys.to(torch::kHalf);
    }
    if (V.scalar_type() != torch::kHalf) {
      V = V.to(torch::kHalf);
    }

    const __half* queries_ptr = reinterpret_cast<const __half*>(queries.data_ptr());
    const __half* keys_ptr = reinterpret_cast<const __half*>(keys.data_ptr());
    __half* attn_scores_ptr = reinterpret_cast<__half*>(attn_scores.data_ptr());

    attention_dot_product_kernel_fp32<__half><<<blocks, threads>>>(
         queries_ptr,
         keys_ptr,
         attn_scores_ptr,
         num_rows, num_cols, head_dim,
         scale
    );

    int threads_softmax = 256;
    softmax_kernel<<<num_rows, threads_softmax, threads_softmax * sizeof(float)>>>(
        reinterpret_cast<__half*>(attn_scores.data_ptr()),
        num_rows, num_cols
    );

    auto output = torch::empty({num_rows, head_dim}, queries.options());

    dim3 blockDim3(16, 16, 1);
    int grid_x3 = (head_dim + WMMA_N - 1) / WMMA_N;
    int grid_y3 = (num_rows + WMMA_M - 1) / WMMA_M;
    dim3 gridDim3(grid_x3, grid_y3);
    attention_weighted_sum_kernel<<<gridDim3, blockDim3>>>(
        reinterpret_cast<__half*>(attn_scores.data_ptr()),
        reinterpret_cast<__half*>(V.data_ptr()),
        reinterpret_cast<__half*>(output.data_ptr()),
        num_rows, num_cols, head_dim
    );

    cudaDeviceSynchronize();
    return {output, attn_scores};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward", &attention_forward, "Attention forward using tensor cores (CUDA)");
}
