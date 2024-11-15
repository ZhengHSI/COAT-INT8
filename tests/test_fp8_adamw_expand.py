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

import math
import sys
import unittest
from dataclasses import dataclass

import torch
import qoptim_cuda

from coat.activation.fake_quantization.quantize_function import (block_cut,
                                                                 block_quant,
                                                                 block_reshape)


@dataclass
class QArgs:
    row_blocksize: int = 1
    col_blocksize: int = 128
    pad_block: bool = True
    first_order_bit: str = "E4M3"
    second_order_bit: str = "E4M3"
    epsilon = 1e-8


qargs = QArgs()

# preparations
beta1, beta2 = 0.9, 0.95
lr, wd, eps = 4e-1, 0.1, 1e-8
step, qgroup_size = 1000, 128
RatioUpperbound = {
    "E4M3": 448 * 448 / 2,
}

M, N = 2, 128
torch.manual_seed(0)
torch.set_printoptions(sci_mode=False)
params = torch.randn((M, N), dtype=torch.float32, device="cuda")
grads = torch.randn((M, N), dtype=torch.float32, device="cuda")
params_py, params_cuda = params.clone(), params.clone()
grads_py, grads_cuda = grads.clone(), grads.clone()

fp_exp_avg = torch.randn((M, N), dtype=torch.float32, device="cuda")
fp_exp_avg_sq = torch.randn((M, N), dtype=torch.float32, device="cuda").abs()

power_options = torch.tensor([i / 16 for i in range(1, 129)]).to(fp_exp_avg)

""" Prepare the data """
# Quantize the first order momentum
Bexp_avg = block_cut(fp_exp_avg, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block)
RQexp_avg, Qexp_avg, Sexp_avg = block_quant(
    Bexp_avg,
    True,
    qargs.first_order_bit,
    stochastic=False,
    epsilon=qargs.epsilon,
    apply_quantize=(qargs.first_order_bit != 100),
    layer_name="first_order",
)
Qexp_avg = block_reshape(Qexp_avg, fp_exp_avg, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block)
fp_exp_avg = block_reshape(RQexp_avg, fp_exp_avg, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block)

# Quantize the second order momentum
Bexp_avg_sq = block_cut(fp_exp_avg_sq, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block)
RQexp_avg_sq, Qexp_avg_sq, Sexp_avg_sq = block_quant(
    Bexp_avg_sq,
    True,
    qargs.first_order_bit,
    stochastic=False,
    epsilon=qargs.epsilon,
    apply_quantize=(qargs.first_order_bit != 100),
    layer_name="first_order",
)
Qexp_avg_sq = block_reshape(
    Qexp_avg_sq, fp_exp_avg_sq, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block
)
fp_exp_avg_sq = block_reshape(
    RQexp_avg_sq, fp_exp_avg_sq, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block
)

Qexp_avg, Qexp_avg_sq = Qexp_avg.to(torch.float8_e4m3fn), Qexp_avg_sq.to(torch.float8_e4m3fn)
Sexp_avg, Sexp_avg_sq = Sexp_avg.to(torch.float32), Sexp_avg_sq.to(torch.float32)

# These are for coat's dynamic range expansion
IDexp_avg = torch.randint(power_options.size(0), size=Sexp_avg.shape)
IDexp_avg_sq = torch.randint(power_options.size(0), size=Sexp_avg_sq.shape)
EXPexp_avg = power_options[IDexp_avg].to(Sexp_avg).squeeze(2)
EXPexp_avg_sq = power_options[IDexp_avg_sq].to(Sexp_avg_sq).squeeze(2)
SQRTexp_avg = torch.randn_like(Sexp_avg).squeeze(2)
SQRTexp_avg_sq = torch.randn_like(Sexp_avg_sq).squeeze(2)

fp_exp_avg = fp_exp_avg.sign() * torch.pow(fp_exp_avg.abs(), 1 / EXPexp_avg) * SQRTexp_avg
fp_exp_avg_sq = torch.pow(fp_exp_avg_sq, 1 / EXPexp_avg_sq) * SQRTexp_avg_sq


def simulated_optimizer_expand_step(params, grads, exp_avg, exp_avg_sq, beta1, beta2, lr, wd, eps, step, qgroup_size):
    # update step
    step += 1

    # Perform stepweight decay
    params.mul_(1 - lr * wd)

    # Decay the first and second moment running average coefficient
    exp_avg.lerp_(grads, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grads, grads, value=1 - beta2)

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    step_size = lr / bias_correction1

    bias_correction2_sqrt = math.sqrt(bias_correction2)

    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

    params.addcdiv_(fp_exp_avg, denom, value=-step_size)

    """ Quantize the first order momentum """
    fp_exp_shape = fp_exp_avg.shape
    q_exp_avg = fp_exp_avg.reshape(1, -1)

    Bexp_avg = block_cut(q_exp_avg, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block)

    # Dynamic Range Expansion
    Bexp_avg_sign = Bexp_avg.sign()
    block_max, block_min = Bexp_avg.abs().max(dim=-1)[0], Bexp_avg.abs().min(dim=-1)[0]
    block_sqrt_minmax1 = (block_max * block_min).sqrt().unsqueeze(2)
    B1Ratio = block_max / block_min

    ratio_upperbound = RatioUpperbound[qargs.first_order_bit]
    B1ExpIdx = (torch.pow(B1Ratio, power_options) < ratio_upperbound).sum(dim=1)
    B1Exp = power_options[B1ExpIdx - 1].unsqueeze(1).unsqueeze(2)

    # re-center for numerical stability
    Bexp_avg = Bexp_avg / block_sqrt_minmax1
    Bexp_avg = torch.pow(Bexp_avg.abs(), B1Exp) * Bexp_avg_sign

    # Dequantize the output to FP32
    RQexp_avg, Qexp_avg, Sexp_avg = block_quant(
        Bexp_avg,
        True,
        qargs.first_order_bit,
        stochastic=False,
        epsilon=qargs.epsilon,
        apply_quantize=(qargs.first_order_bit != 100),
        layer_name="first_order",
    )
    Qexp_avg = block_reshape(Qexp_avg, q_exp_avg, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block)

    RQexp_avg = torch.pow(RQexp_avg.abs(), 1 / B1Exp) * Bexp_avg_sign
    RQexp_avg = RQexp_avg * block_sqrt_minmax1

    RQexp_avg = block_reshape(RQexp_avg, q_exp_avg, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block)
    rq_exp_avg = RQexp_avg.reshape(fp_exp_shape)

    fp_exp_avg.data = rq_exp_avg

    """ Quantize the second order momentum """
    fp_exp_shape = fp_exp_avg_sq.shape
    q_exp_avg_sq = fp_exp_avg_sq.reshape(1, -1)
    Bexp_avg_sq = block_cut(q_exp_avg_sq, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block)

    # Dynamic Range Expansion
    block_max, block_min = Bexp_avg_sq.max(dim=-1)[0], Bexp_avg_sq.min(dim=-1)[0]
    block_sqrt_minmax2 = (block_max * block_min).sqrt().unsqueeze(2)
    B2Ratio = block_max / block_min

    ratio_upperbound = RatioUpperbound[qargs.second_order_bit]
    B2ExpIdx = (torch.pow(B2Ratio, power_options) < ratio_upperbound).sum(dim=1)
    B2Exp = power_options[B2ExpIdx - 1].unsqueeze(1).unsqueeze(2)

    # re-center for numerical stability
    Bexp_avg_sq = Bexp_avg_sq / block_sqrt_minmax2
    Bexp_avg_sq = torch.pow(Bexp_avg_sq, B2Exp)

    # Dequantize the output to FP32
    RQexp_avg_sq, Qexp_avg_sq, Sexp_avg_sq = block_quant(
        Bexp_avg_sq,
        True,
        qargs.second_order_bit,
        stochastic=False,
        epsilon=qargs.epsilon,
        apply_quantize=(qargs.second_order_bit != 100),
        layer_name="second_order",
    )
    Qexp_avg_sq = block_reshape(
        Qexp_avg_sq, q_exp_avg_sq, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block
    )

    RQexp_avg_sq = torch.pow(RQexp_avg_sq.abs(), 1 / B2Exp)
    RQexp_avg_sq = RQexp_avg_sq * block_sqrt_minmax2

    RQexp_avg_sq = block_reshape(
        RQexp_avg_sq, q_exp_avg_sq, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block
    )
    rq_exp_avg_sq = RQexp_avg_sq.reshape(fp_exp_shape)

    fp_exp_avg_sq.data = rq_exp_avg_sq

    return Qexp_avg, Sexp_avg, B1Exp, block_sqrt_minmax1, Qexp_avg_sq, Sexp_avg_sq, B2Exp, block_sqrt_minmax2


def cuda_optimizer_expand_step(
    params,
    grads,
    exp_avg,
    scale_exp_avg,
    expand_exp_avg,
    sqrt_minmax_exp_avg,
    exp_avg_sq,
    scale_exp_avg_sq,
    expand_exp_avg_sq,
    sqrt_minmax_exp_avg_sq,
    beta1,
    beta2,
    lr,
    wd,
    eps,
    step,
    qgroup_size,
    expand_min,
):
    qoptim_cuda.fp8_adamw_expand_step(
        params,
        grads,
        exp_avg,
        scale_exp_avg,
        expand_exp_avg,
        sqrt_minmax_exp_avg,
        exp_avg_sq,
        scale_exp_avg_sq,
        expand_exp_avg_sq,
        sqrt_minmax_exp_avg_sq,
        beta1,
        beta2,
        lr,
        wd,
        eps,
        step,
        qgroup_size,
        expand_min,
    )


class TestExpandAdamW(unittest.TestCase):
    def test_expand_coatadamw(self):
        (
            Qexp_avg_py,
            Sexp_avg_py,
            B1Exp,
            block_sqrt_minmax1,
            Qexp_avg_sq_py,
            Sexp_avg_sq_py,
            B2Exp,
            block_sqrt_minmax2,
        ) = simulated_optimizer_expand_step(
            params_py, grads_py, fp_exp_avg, fp_exp_avg_sq, beta1, beta2, lr, wd, eps, step, qgroup_size
        )

        cuda_optimizer_expand_step(
            params_cuda,
            grads_cuda,
            Qexp_avg,
            Sexp_avg,
            EXPexp_avg,
            SQRTexp_avg,
            Qexp_avg_sq,
            Sexp_avg_sq,
            EXPexp_avg_sq,
            SQRTexp_avg_sq,
            beta1,
            beta2,
            lr,
            wd,
            eps,
            step,
            qgroup_size,
            16,
        )

        self.assertTrue(torch.allclose(params_py.flatten(), params_cuda.flatten(), 1e-4))
        self.assertTrue(
            torch.allclose(Qexp_avg_py.to(torch.bfloat16).flatten(), Qexp_avg.to(torch.bfloat16).flatten(), 1e-4)
        )
        self.assertTrue(torch.allclose(Sexp_avg_py.flatten(), Sexp_avg.flatten(), 1e-4))
        self.assertTrue(torch.allclose(B1Exp.flatten(), EXPexp_avg.flatten(), 1e-4))
        self.assertTrue(torch.allclose(block_sqrt_minmax1.flatten(), SQRTexp_avg.flatten(), 1e-4))
        self.assertTrue(
            torch.allclose(Qexp_avg_sq_py.to(torch.bfloat16).flatten(), Qexp_avg_sq.to(torch.bfloat16).flatten(), 1e-4)
        )
        self.assertTrue(torch.allclose(Sexp_avg_sq_py.flatten(), Sexp_avg_sq.flatten(), 1e-4))
        self.assertTrue(torch.allclose(B2Exp.flatten(), EXPexp_avg_sq.flatten(), 1e-4))
        self.assertTrue(torch.allclose(block_sqrt_minmax2.flatten(), SQRTexp_avg_sq.flatten(), 1e-4))


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False, linewidth=200)
    unittest.main()
