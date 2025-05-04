#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>
#include <cuda_bf16.h>

template <typename scalar_t>
__global__ void block_diagonal_mm_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    int num_blocks,
    int block_size,
    int total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int block_idx = idx / block_size;
    int pos_in_block = idx % block_size;
    int block_row = block_idx % num_blocks;

    scalar_t sum = 0;
    for (int k = 0; k < block_size; k++) {
        int input_idx = block_idx * block_size + k;
        int weight_idx = block_row * block_size * block_size + k * block_size + pos_in_block;
        sum += input[input_idx] * weight[weight_idx];
    }
    output[idx] = sum;
}

torch::Tensor block_diagonal_mm_cuda(
    torch::Tensor input,
    torch::Tensor weight) {

    int num_blocks = weight.size(0);
    int block_size = weight.size(1);
    auto output = torch::zeros_like(input);
    int total_elements = input.numel();

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "block_diagonal_mm_cuda",
        ([&] {
            block_diagonal_mm_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                num_blocks,
                block_size,
                total_elements);
        })
    );

    return output;
}

template <typename scalar_t>
__global__ void diagonal_mv_kernel(
    const scalar_t* matrix,
    const scalar_t* vector,
    scalar_t* output,
    int batch_size,
    int seq_len,
    int dim) {
    
    int batch = blockIdx.x;
    int seq = blockIdx.y;
    int idx = threadIdx.x;

    if (batch >= batch_size || seq >= seq_len || idx >= dim) return;

    int matrix_idx = batch * seq_len * dim + seq * dim + idx;
    int vector_idx = batch * seq_len * dim + seq * dim + idx;
    
    output[matrix_idx] = matrix[matrix_idx] * vector[vector_idx];
}

torch::Tensor diagonal_mv_cuda(
    torch::Tensor matrix,
    torch::Tensor vector) {

    auto output = torch::zeros_like(matrix);
    int batch_size = matrix.size(0);
    int seq_len = matrix.size(1);
    int dim = matrix.size(2);

    dim3 blocks(batch_size, seq_len);
    dim3 threads(dim);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        matrix.scalar_type(),
        "diagonal_mv_cuda",
        ([&] {
            diagonal_mv_kernel<scalar_t><<<blocks, threads>>>(
                matrix.data_ptr<scalar_t>(),
                vector.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                seq_len,
                dim);
        })
    );

    return output;
}

template <typename scalar_t>
__global__ void rnn_scan_kernel(
    const scalar_t* x,
    const scalar_t* a,
    const bool* reset,
    scalar_t* y,
    const scalar_t* h0,
    scalar_t* last_h,
    int batch_size,
    int seq_len,
    int features) {

    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    int feature = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch >= batch_size || feature >= features) return;

    scalar_t h = h0 ? h0[batch * features + feature] : scalar_t(0);
    int reset_idx = batch * seq_len;

    for (int t = 0; t < seq_len; ++t) {
        if (reset[reset_idx + t]) {
            h = h0 ? h0[batch * features + feature] : scalar_t(0);
        }
        int idx = (batch * seq_len + t) * features + feature;
        h = a[idx] * h + x[idx];
        y[idx] = h;
    }
    last_h[batch * features + feature] = h;
}

std::vector<torch::Tensor> rnn_scan_cuda_forward(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor reset,
    torch::Tensor h0) {

    auto y = torch::zeros_like(x);
    auto last_h = torch::empty({x.size(0), x.size(2)}, x.options());

    dim3 blocks(
        (x.size(0) + 31) / 32,
        (x.size(2) + 31) / 32
    );
    dim3 threads(32, 32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "rnn_scan_cuda",
        ([&] {
            rnn_scan_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                a.data_ptr<scalar_t>(),
                reset.data_ptr<bool>(),
                y.data_ptr<scalar_t>(),
                h0.defined() ? h0.data_ptr<scalar_t>() : nullptr,
                last_h.data_ptr<scalar_t>(),
                x.size(0),
                x.size(1),
                x.size(2));
        })
    );

    return {y, last_h};
}

// Kernel for RMSNorm forward pass using FP32 accumulation
template <typename scalar_t>
__global__ void rms_norm_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ scale,
    scalar_t* __restrict__ output,
    int N,
    int d,
    float eps) {

    extern __shared__ float shared[];
    int row = blockIdx.x;
    int lane = threadIdx.x;
    if (row >= N) return;
    float sum = 0.0f;
    for (int j = lane; j < d; j += blockDim.x) {
        float val = static_cast<float>(input[row * d + j]);
        sum += val * val;
    }
    shared[lane] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            shared[lane] += shared[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        float rms = sqrtf(shared[0] / d + eps);
        for (int j = 0; j < d; j++) {
            float val = static_cast<float>(input[row * d + j]);
            float scale_val = static_cast<float>(scale[j]);
            float out_val = (val / rms) * (scale_val + 1.0f);
            output[row * d + j] = static_cast<scalar_t>(out_val);
        }
    }
}

std::vector<torch::Tensor> rms_norm_forward(
    torch::Tensor input,
    torch::Tensor scale,
    float eps) {

    auto sizes = input.sizes();
    int d = sizes[sizes.size() - 1];
    int N = input.numel() / d;
    auto input_2d = input.view({N, d});
    auto output = torch::empty_like(input_2d);

    int threads = min(1024, d);
    int blocks = N;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rms_norm_forward_cuda", ([&] {
            rms_norm_forward_kernel<scalar_t><<<blocks, threads, threads * sizeof(float)>>>(
                input.data_ptr<scalar_t>(),
                scale.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                N, d, eps
            );
        })
    );

    return {output.view_as(input)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rnn_scan_cuda_forward", &rnn_scan_cuda_forward, "Optimized temporal_rnn_scan (CUDA)");
    m.def("rms_norm_forward", &rms_norm_forward, "Optimized rms_norm_forward (CUDA)");
    m.def("block_diagonal_mm", &block_diagonal_mm_cuda, "Block diagonal matrix multiplication (CUDA)");
    m.def("diagonal_mv", &diagonal_mv_cuda, "Block diagonal matrix multiplication (CUDA)");
}
