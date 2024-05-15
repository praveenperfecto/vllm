#pragma once
#include <torch/extension.h>

// Declare CUDA functions
void silu_and_mul(torch::Tensor& out, const torch::Tensor& input);
void gelu_and_mul(torch::Tensor& out, const torch::Tensor& input);
void gelu_tanh_and_mul(torch::Tensor& out, const torch::Tensor& input);
