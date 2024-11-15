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

# This repo is used to reproduce Table 6 in COAT paper

import pandas as pd
import torch
import argparse
from transformers import AutoConfig, AutoModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from coat.activation.models._fp8manager import FP8Manager
from coat.activation.models._fp8_quantization_config import QuantizationConfig
from coat.activation.models.coat_llama import CoatLlamaDecoderLayer, CoatLlamaModel, make_state_dict_compatible
from coat.activation.real_quantization import (Coat_quantize_bgn,
                                               Coat_quantize_end)

torch.manual_seed(0)
batch_sizes = [8]
sequence_lengths = [2048]
n_repeat = 20

torch.empty(
    1, device="cuda", requires_grad=True
).backward()  # Triton will throw RuntimeError: Triton Error [CUDA]: invalid device context if you comment this line

config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")
config._attn_implementation = "flash_attention_2"
model = AutoModel.from_pretrained("meta-llama/Llama-3.1-8B")

def tester(CHOICE):
    if CHOICE == "BF16":
        llama_layer = LlamaDecoderLayer(config=config, layer_idx=0)
        llama_layer = llama_layer.cuda().to(torch.bfloat16)
        llama_layer.load_state_dict(model.layers[0].state_dict())

        # Forward Time Benchmark
        for batch_size in batch_sizes:
            for sequence_length in sequence_lengths:
                dummy_input = (
                    torch.rand((batch_size, sequence_length, config.hidden_size), dtype=torch.bfloat16)
                    .cuda()
                    .requires_grad_(True)
                )

                attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.bool).cuda()
                position_ids = torch.arange(sequence_length, dtype=torch.long, device="cuda").unsqueeze(0).expand(batch_size, -1)

                dummy_grad = torch.rand((batch_size, sequence_length, config.hidden_size), dtype=torch.bfloat16).cuda()

                # Test Backward Time
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    dummy_output = llama_layer(dummy_input, attention_mask, position_ids)
                    dummy_output[0].backward(dummy_grad)
                    torch.cuda.synchronize()
                    import IPython
                    IPython.embed()

    if CHOICE == "COAT":
        fp8_config = QuantizationConfig(
            quantize_model=True,
            group_size=16,
        )
        config.coat_fp8_args = fp8_config

        llama_layer = CoatLlamaDecoderLayer(config, layer_idx=0)
        llama_layer.load_state_dict(make_state_dict_compatible(model.layers[0].state_dict()))

        llama_bgn = Coat_quantize_bgn(config.coat_fp8_args).cuda()
        llama_end = Coat_quantize_end(config.coat_fp8_args).cuda()
        llama_layer = llama_layer.cuda().to(torch.bfloat16)

        # Forward Time Benchmark
        for batch_size in batch_sizes:
            for sequence_length in sequence_lengths:
                dummy_input = (
                    torch.rand((batch_size, sequence_length, config.hidden_size), dtype=torch.bfloat16)
                    .cuda()
                    .requires_grad_(True)
                )
                
                attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.bool).cuda()
                position_ids = torch.arange(sequence_length, dtype=torch.long, device="cuda").unsqueeze(0).expand(batch_size, -1)

                dummy_grad = torch.rand((batch_size, sequence_length, config.hidden_size), dtype=torch.bfloat16).cuda()

                dummy_x, dummy_qx, dummy_sx = llama_bgn(dummy_input)

                # Test Backward Time
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    FP8Manager.is_first_microbatch = True
                    dummy_output_x, dummy_output_qx, dummy_output_sx = llama_layer(
                        dummy_x, dummy_qx, dummy_sx, attention_mask, position_ids
                    )
                    dummy_output = llama_end(dummy_output_x, dummy_output_qx, dummy_output_sx)
                    dummy_output.backward(dummy_grad)
                    torch.cuda.synchronize()
                    import IPython
                    IPython.embed()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Set the method for training.")
    parser.add_argument(
        "--method",
        type=str,
        choices=["COAT", "BF16"],
        required=True,
        help="Choose the training method: 'COAT' or 'BF16'."
    )
    return parser.parse_args()

torch.manual_seed(0)
torch.set_printoptions(linewidth=200, sci_mode=False)
args = parse_arguments()
tester(args.method)