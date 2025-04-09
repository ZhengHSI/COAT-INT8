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

import re

import torch

from .FloatPointQuantizeTorch import *
from .FloatPointQuantizeTriton import *


def block_cut(input, row_block, column_block, pad_block=False):
    # print(input.shape)
    original_shape = input.shape
    # input tensor shape is M * N
    if len(input.shape) > 2:
        input = input.reshape(-1, input.shape[2])
    elif len(input.shape) == 2:
        pass
    else:
        raise ValueError(f"input shape {input.shape} does not match for block cut, {input}")
    M, N = input.shape[0], input.shape[1]

    if row_block == -1:
        row_block = M
    if column_block == -1:
        column_block = N

    if pad_block:
        row_remainder, col_remainder = M % row_block, N % column_block
        if row_remainder:
            row_pad = row_block - row_remainder
        else:
            row_pad = 0
        if col_remainder:
            col_pad = column_block - col_remainder
        else:
            col_pad = 0

        input = torch.nn.functional.pad(
            input, (0, col_pad, 0, row_pad), "constant", 0
        )  # refer to torch's doc to see why
        M, N = input.shape[0], input.shape[1]
        row_num, column_num = M // row_block, N // column_block
    else:
        row_num, column_num = M // row_block, N // column_block

    assert row_num * row_block == M, f"{row_num}, {row_block}, {M}, {original_shape}"
    assert column_num * column_block == N, f"{column_num}, {column_block}, {N}, {original_shape}"
    # print(input.shape)
    input = (
        input.reshape(row_num, row_block, column_num, column_block)
        .permute(0, 2, 1, 3)
        .reshape(row_num * column_num, row_block, column_block)
    )
    # print(input.shape)
    return input


def block_reshape(input, origin_input, row_block, column_block, pad_block=False):
    if len(origin_input.shape) > 2:
        flatten_input = origin_input.reshape(-1, origin_input.shape[2])
    elif len(origin_input.shape) == 2:
        flatten_input = origin_input
    else:
        raise ValueError(f"input shape {input.shape} does not match for block cut")

    M, N = flatten_input.shape[0], flatten_input.shape[1]

    if row_block == -1:
        row_block = M
    if column_block == -1:
        column_block = N

    if pad_block:
        row_remainder, col_remainder = M % row_block, N % column_block
        if row_remainder:
            row_pad = row_block - row_remainder
        else:
            row_pad = 0
        if col_remainder:
            col_pad = column_block - col_remainder
        else:
            col_pad = 0

        pad_origin_input = torch.nn.functional.pad(origin_input, (0, col_pad, 0, row_pad), "constant", 0)
        M, N = pad_origin_input.shape[0], pad_origin_input.shape[1]
        row_num, column_num = M // row_block, N // column_block
    else:
        row_num, column_num = M // row_block, N // column_block

    input = (
        input.reshape(row_num, column_num, row_block, column_block)
        .permute(0, 2, 1, 3)
        .reshape(row_num * row_block, column_num * column_block)
    )

    M, N = flatten_input.shape[0], flatten_input.shape[1]
    input = input[:M, :N]

    if len(origin_input.shape) > 2:
        input = input.reshape(origin_input.shape)
    elif len(origin_input.shape) == 2:
        pass
    else:
        raise ValueError(f"input shape {input.shape} does not match for block reshape")

    return input


def block_verify_int8(input, row_block, column_block, layer_type, necessary=True):
    Binput = block_cut(input, row_block, column_block)
    Binput = Binput.to(torch.float32)

    for n in range(Binput.shape[0]):
        unique_values = len(torch.unique(Binput[n, :, :]))
        if unique_values > 256:
            if necessary:
                raise ValueError(f"{layer_type} contains more than 256 unique values.")
            else:
                return False
    return True


def block_quant(input, symm, bits, stochastic, epsilon, apply_quantize, layer_name):
    Quant_fn = SymmQuantizer
    return Quant_fn.apply(input, symm, bits, stochastic, epsilon, apply_quantize, layer_name)


def extract_bit(string):
    match_int = re.match(r"INT(\d+)", string)  # INT8
    match_mx_fp = re.match(r"MXE(\d+)M(\d+)", string)  # MXFP4 format
    match_mx_fp_plus = re.match(r"MXE(\d+)M(\d+)_plus", string)  # MXFP4_plus format
    match_nv_fp = re.match(r"NVE(\d+)M(\d+)", string)  # NVFP4 format
    match_nv_fp_plus = re.match(r"NVE(\d+)M(\d+)_plus", string)  # NVFP4_plus format
    match_float = re.match(r"E(\d+)M(\d+)", string)  # E4M3 / E5M2
    
    # Avoid double match. Plus is higher priority
    if match_mx_fp_plus:
        match_mx_fp = None
    if match_nv_fp_plus:
        match_nv_fp = None
    
    if match_int:
        return "integer", int(match_int.group(1)), None
    elif match_float:
        Ebit, Mbit = int(match_float.group(1)), int(match_float.group(2))
        if Ebit == 1:
            return "integer", Mbit + 1, None
        if Mbit == 0:
            return "floatExM0", Ebit, 0
        return "floatExMy", Ebit, Mbit
    elif match_mx_fp:
        Ebit, Mbit = int(match_mx_fp.group(1)), int(match_mx_fp.group(2))
        return "MXExMy", Ebit, Mbit
    elif match_mx_fp_plus:
        Ebit, Mbit = int(match_mx_fp_plus.group(1)), int(match_mx_fp_plus.group(2))
        return "MXExMy_plus", Ebit, Mbit
    elif match_nv_fp:
        Ebit, Mbit = int(match_nv_fp.group(1)), int(match_nv_fp.group(2))
        return "NVExMy", Ebit, Mbit
    elif match_nv_fp_plus:
        Ebit, Mbit = int(match_nv_fp_plus.group(1)), int(match_nv_fp_plus.group(2))
        return "NVExMy_plus", Ebit, Mbit
    raise ValueError(f"{string} data format is not supported")


def find_max_min(input, quant_type, absmax_per_block, bit1, bit2, symm = False, block_size = 32):

    if quant_type == "integer":
        Qn, Qp = -(2 ** (bit1 - 1) - 1), 2 ** (bit1 - 1) - 1
    elif quant_type in ["floatExMy", "MXExMy", "MXExMy_plus", "NVExMy", "NVExMy_plus"]:
        Qp = (2 - 2 ** (-bit2)) * (2 ** (2 ** (bit1 - 1)))
        if bit1 == 4 and bit2 == 3:
            # https://arxiv.org/pdf/2209.05433: Force it to be 448, as the principle is different
            Qp = 448
        if bit1 == 5 and bit2 == 2:
            # https://arxiv.org/pdf/2209.05433: Force it to be 57344, as the principle is different
            Qp = 57344 
        Qn = -Qp
    elif quant_type == "floatExM0":
        Qp = 2 ** (2 ** (bit1 - 1) - 1)
        Qn = -Qp
    else:
        raise NotImplementedError(f"{bit1} & {bit2} is not supported by quantization")
    
    scale_per_block = (2 * absmax_per_block) / (Qp - Qn)
    scale_per_block = scale_per_block.to(input)
    
    if quant_type == "MXExMy":
        scale_per_block = floatExM0_quantize_torch(scale_per_block, 8, stochastic=False, ceil=True) # Scaling Factor should be E8M0
    elif quant_type == "MXExMy_plus":
        # Scaling factor should be E8M0
        double_scale = scale_per_block.abs().max() / torch.tensor(2 ** 127, dtype=torch.float32, device=scale_per_block.device)
        scale_per_block = floatExM0_quantize_torch(scale_per_block / double_scale, 8, stochastic=False, ceil=True) # Scaling Factor should be E8M0
        scale_per_block = scale_per_block * double_scale
    elif quant_type == "NVExMy":
        scale_per_block = floatExMy_quantize_torch(scale_per_block, 4, 3, stochastic=False, ceil=True) # Scaling Factor should be E4M3
    elif quant_type == "NVExMy_plus":
        # Scaling factor should be E4M3
        double_scale = scale_per_block.abs().max() / 448
        scale_per_block = floatExMy_quantize_torch(scale_per_block / double_scale, 4, 3, stochastic=False, ceil=True) # Scaling Factor should be E4M3
        scale_per_block = scale_per_block * double_scale
        
    return scale_per_block, Qn, Qp
    
class SymmQuantizer(torch.autograd.function.InplaceFunction):
    @staticmethod
    def forward(ctx, input, symm, bits, stochastic, epsilon, apply_quantize=True, layer_name=None):
        with torch.no_grad():
            absmax_per_block = input.abs().amax(dim=(1, 2)).unsqueeze(1).unsqueeze(2) + epsilon

            if bits == "100" or not apply_quantize:
                return input, input, torch.ones_like(absmax_per_block)
            elif bits == "FP32":
                return input.to(torch.float32), input.to(torch.float32), torch.ones_like(absmax_per_block)
            elif bits == "FP16":
                return input.to(torch.float16), input.to(torch.float16), torch.ones_like(absmax_per_block)
            elif bits == "BF16":
                return input.to(torch.bfloat16), input.to(torch.bfloat16), torch.ones_like(absmax_per_block)
            else:
                QuantType, bit1, bit2 = extract_bit(bits)
                if not symm:
                    bit1 = bit1 + 1  # pretend to be asymmtric

                scale_per_block, Qn, Qp = find_max_min(input, QuantType, absmax_per_block, bit1, bit2, symm)
                Qinput = input / scale_per_block

                if QuantType == "integer":
                    if stochastic:
                        noise = Qinput.new(Qinput.shape).uniform_(-0.5, 0.5)
                        Qinput.add_(noise)
                    Qinput.clamp_(Qn, Qp).round_()
                elif QuantType in ["floatExMy", "MXExMy", "MXExMy_plus", "NVExMy", "NVExMy_plus"]:
                    # Qinput = floatExMy_quantize_torch(Qinput, bit1, bit2, stochastic)
                    Qinput = floatExMy_quantize_triton(Qinput, bit1, bit2, stochastic)
                elif QuantType == "floatExM0":
                    Qinput = floatExM0_quantize_torch(Qinput, bit1, stochastic)
                else:
                    raise NotImplementedError(f"{bits} is not supported by quantization")

                RQinput = Qinput * scale_per_block

                if input.dtype != Qinput.dtype:
                    print(
                        f"Input type is {input.dtype}, Qinput type is {Qinput.dtype}, scale_per_block type is {scale_per_block.dtype}",
                        file=open("debug.txt", "a"),
                    )
                    import IPython
                    IPython.embed()
                return RQinput, Qinput, scale_per_block

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None
