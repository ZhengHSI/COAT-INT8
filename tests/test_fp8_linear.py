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

import unittest

import torch
from test_utils import check_similarity, dequantize_tensor, quantize_tensor

from coat.activation.real_quantization.linear import fp8matmul


class TestLinear(unittest.TestCase):
    def test_linear(self):
        M, N, K = 4096, 2048, 8192
        QB = 16
        a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)

        scale_a, scale_b = torch.randn((1), device="cuda", dtype=torch.bfloat16), torch.randn(
            (1), device="cuda", dtype=torch.bfloat16
        )

        # Prepare row major and col-major data
        a = a.to(torch.float8_e4m3fn)
        b = b.T
        b = b.to(torch.float8_e4m3fn)

        a_32, b_32 = a.to(torch.float32), b.to(torch.float32)
        scale_ab = scale_a.to(torch.float32) * scale_b.to(torch.float32)
        output_torch = torch.matmul(a_32, b_32) * scale_ab
        output_torch_quantized, _, __ = quantize_tensor(output_torch.unsqueeze(0), 1, M, N, QB, torch.float8_e4m3fn)
        output_torch = output_torch.to(torch.bfloat16)

        # Output is not quantized
        output_fp8 = fp8matmul(a, b, False, scale_a, scale_b, QB)  # a should be row-major, b should be col-major

        # Output is quantized
        output_fp8_qx, output_fp8_sx = fp8matmul(
            a, b, True, scale_a, scale_b, QB
        )  # a should be row-major, b should be col-major
        output_fp8_rqx = dequantize_tensor(
            output_fp8_qx.unsqueeze(0), output_fp8_sx.unsqueeze(0), 1, M, N, QB
        )  # dequantize per-tensor quantization

        # TODO: This is not ideal
        self.assertTrue(torch.allclose(output_torch, output_fp8, 1e-2, 0.2))
        self.assertTrue(torch.allclose(output_torch_quantized, output_fp8_rqx, 1e-1, 0.5))


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False, linewidth=200)
    unittest.main()
