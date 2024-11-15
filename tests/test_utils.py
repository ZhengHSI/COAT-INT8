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

from coat.activation.real_quantization.common import FP8_MAX_VALUE


def random_tensor_generator(BS, SL, CDIM, QB, fp8type):
    """
    Generates a random tensor, quantizes it, and returns the quantized result.

    Parameters:
    - BS (int): Batch size.
    - SL (int): Sequence length.
    - CDIM (int): Channel dimension (input dimension).
    - QB (int): Quantization block size. Should be the power of 2.
    - fp8type (torch.dtype): Target data type for quantization (e.g., torch.float8_e4m3fn, torch.float8_e5m2).

    Returns:
    - rqx (torch.Tensor): Reconstructed tensor after quantization. Should be equal to qx * sx
    - qx (torch.Tensor): Quantized tensor.
    - sx (torch.Tensor): Scaling factors used in quantization.
    """
    x = torch.randn(BS, SL, CDIM).cuda()
    rqx, qx, sx = quantize_tensor(x, BS, SL, CDIM, QB, fp8type)
    return rqx, qx, sx


def quantize_tensor(x, BS, SL, CDIM, QB, fp8type, per_tensor=False):
    """
    Quantizes a given tensor using a per-group quantization strategy, converting it into fp8 format.
    """
    if per_tensor:
        sx = (x.abs().max() / FP8_MAX_VALUE[fp8type]).to(torch.bfloat16)
        qx = (x / sx).to(fp8type)
        rqx = qx.to(torch.float32) * sx.to(torch.float32)
    else:
        _qx = x.reshape(BS, SL, CDIM // QB, QB)
        sx = _qx.abs().amax(dim=(3)) / FP8_MAX_VALUE[fp8type]
        sx = sx.to(torch.bfloat16)
        _qx = (_qx / sx.unsqueeze(3)).to(fp8type)
        qx = _qx.reshape(BS, SL, CDIM)
        rqx = (_qx.float() * sx.unsqueeze(3).float()).reshape(BS, SL, CDIM)
    return rqx, qx, sx


def dequantize_tensor(x_triton, s_triton, BS, SL, CDIM, QB):
    """
    Dequantizes a tensor by converting it back to full precision using scaling factors.
    """
    if s_triton.numel() == 1:
        # In this case it should be per-tensor quantization
        output_triton = x_triton.to(torch.float32) * s_triton.to(torch.float32)
    else:
        # In this case it should be per-group quantization with group size 1 * G
        assert len(s_triton.shape) == 3
        _x_triton = x_triton.reshape(BS, SL, CDIM // QB, QB)
        _x_triton = _x_triton.to(torch.float32)
        s_triton = s_triton.unsqueeze(3).to(torch.float32)
        output_triton = (_x_triton * s_triton).reshape(BS, SL, CDIM)
    return output_triton


def check_similarity(output_torch, output_triton, threshold=0.95):
    """
    Check if at least a specified percentage (default 95%) of values in two tensors are exactly the same.

    Parameters:
    - output_torch (torch.Tensor): First tensor to compare.
    - output_triton (torch.Tensor): Second tensor to compare.
    - threshold (float): The percentage of matching elements required (default is 99%).

    Returns:
    - bool: True if the percentage of matching elements is above the threshold, False otherwise.
    """
    # Compare the two tensors element-wise for equality
    equal_elements = torch.eq(output_torch, output_triton)

    # Calculate the ratio of equal elements
    num_equal_elements = torch.sum(equal_elements).item()
    total_elements = output_torch.numel()
    similarity_ratio = num_equal_elements / total_elements

    # Check if the similarity ratio meets the threshold
    return similarity_ratio >= threshold
