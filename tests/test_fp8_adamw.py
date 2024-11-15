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

M, N = 1, 127
torch.manual_seed(0)
torch.set_printoptions(sci_mode=False)
params = torch.randn((M, N), dtype=torch.float32, device="cuda")
grads = torch.randn((M, N), dtype=torch.float32, device="cuda")
params_py, params_cuda = params.clone(), params.clone()
grads_py, grads_cuda = grads.clone(), grads.clone()

fp_exp_avg = torch.randn((M, N), dtype=torch.float32, device="cuda")
fp_exp_avg_sq = torch.randn((M, N), dtype=torch.float32, device="cuda").abs()

# Simulation Quantization
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

Bexp_avg_sq = block_cut(fp_exp_avg_sq, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block)
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
    Qexp_avg_sq, fp_exp_avg_sq, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block
)
fp_exp_avg_sq = block_reshape(
    RQexp_avg_sq, fp_exp_avg_sq, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block
)

Qexp_avg, Qexp_avg_sq = Qexp_avg.to(torch.float8_e4m3fn), Qexp_avg_sq.to(torch.float8_e4m3fn)
Sexp_avg, Sexp_avg_sq = Sexp_avg.to(torch.float32), Sexp_avg_sq.to(torch.float32)


def python_optimizer_step(params, grads, exp_avg, exp_avg_sq, beta1, beta2, lr, wd, eps, step, qgroup_size):
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

    # quantize the first order momentum
    fp_exp_shape = fp_exp_avg.shape
    q_exp_avg = fp_exp_avg.reshape(1, -1)
    Bexp_avg = block_cut(q_exp_avg, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block)
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
    RQexp_avg = block_reshape(RQexp_avg, q_exp_avg, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block)
    rq_exp_avg = RQexp_avg.reshape(fp_exp_shape)

    fp_exp_avg.data = rq_exp_avg

    # quantize the second order momentum
    fp_exp_shape = fp_exp_avg_sq.shape
    q_exp_avg_sq = fp_exp_avg_sq.reshape(1, -1)
    Bexp_avg_sq = block_cut(q_exp_avg_sq, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block)
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
    RQexp_avg_sq = block_reshape(
        RQexp_avg_sq, q_exp_avg_sq, qargs.row_blocksize, qargs.col_blocksize, pad_block=qargs.pad_block
    )
    rq_exp_avg_sq = RQexp_avg_sq.reshape(fp_exp_shape)

    fp_exp_avg_sq.data = rq_exp_avg_sq

    return Qexp_avg, Sexp_avg, Qexp_avg_sq, Sexp_avg_sq


def cuda_optimizer_step(
    params, grads, exp_avg, scale_exp_avg, exp_avg_sq, scale_exp_avg_sq, beta1, beta2, lr, wd, eps, step, qgroup_size
):
    qoptim_cuda.fp8_adamw_step(
        params,
        grads,
        exp_avg,
        scale_exp_avg,
        exp_avg_sq,
        scale_exp_avg_sq,
        beta1,
        beta2,
        lr,
        wd,
        eps,
        step,
        qgroup_size,
    )


# import IPython
# IPython.embed()
Qexp_avg_py, Sexp_avg_py, Qexp_avg_sq_py, Sexp_avg_sq_py = python_optimizer_step(
    params_py, grads_py, fp_exp_avg, fp_exp_avg_sq, beta1, beta2, lr, wd, eps, step, qgroup_size
)
# import IPython
# IPython.embed()
cuda_optimizer_step(
    params_cuda, grads_cuda, Qexp_avg, Sexp_avg, Qexp_avg_sq, Sexp_avg_sq, beta1, beta2, lr, wd, eps, step, qgroup_size
)
torch.cuda.synchronize()
import IPython

IPython.embed()
