# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton

from coat.activation.real_quantization.linear import fp8matmul


def benchmarker(M, N, K, provider, groupsize: int = 16):

    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((N), device="cuda", dtype=torch.bfloat16)

    a16, b16 = a.to(torch.bfloat16), b.to(torch.bfloat16)

    scale_a, scale_b = torch.randn((1), device="cuda", dtype=torch.bfloat16), torch.randn(
        (1), device="cuda", dtype=torch.bfloat16
    )
    a = a.to(torch.float8_e4m3fn)
    b = b.T
    b = b.to(torch.float8_e4m3fn)

    def torch_bf16():
        output_bf16 = torch.matmul(a16, b16)

    def triton_fp8_output_fp():
        output_fp8_y = fp8matmul(a, b, False, scale_a, scale_b, groupsize, bias=bias)

    def triton_fp8_output_quantized():
        output_fp8_y, output_fp8_s = fp8matmul(a, b, True, scale_a, scale_b, groupsize, bias=bias)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch-bf16":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_bf16(), quantiles=quantiles)
    elif provider == "triton-fp8-output-fp":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_fp8_output_fp(), quantiles=quantiles)
    elif provider == "triton-fp8-output-quantized":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_fp8_output_quantized(), quantiles=quantiles)

    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=[512 * i for i in range(2, 17)],
            line_arg="provider",
            line_vals=["torch-bf16", "triton-fp8-output-fp", "triton-fp8-output-quantized"],
            line_names=["Torch-BF16", "Triton-FP8-Output-FP", "Triton-FP8-Output-Quantized"],
            styles=[("green", "-"), ("blue", "-"), ("red", "-")],
            ylabel="TFLOPS",
            plot_name="matmul-performance",
            args={},
        )
    )

    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, provider):
        return benchmarker(M, N, K, provider)

    # def benchmark(M, provider):
    #     return benchmarker(M, 4096, 4096, 128, provider)

    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False, linewidth=200, precision=6)
    benchmark.run(print_data=True)
