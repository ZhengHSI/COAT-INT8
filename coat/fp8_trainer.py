# In this file I only add the logic in line 453 to 456. The rest remains unchanged compared with the original Trainer Class.

import os
import math
import time
import torch
import shutil
from packaging import version
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch.nn as nn

from transformers import Trainer
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from transformers.trainer_pt_utils import get_model_param_count
from transformers.integrations import hp_params
from transformers.trainer_utils import (
    has_length,
    speed_metrics,
    HPSearchBackend,
    TrainOutput
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.trainer_callback import (
    TrainerState,
    ExportableState,
)
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.utils import (
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_in_notebook,
    is_ipex_available,
    is_liger_kernel_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    is_torch_compile_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging
)

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration


from .utils._fp8manager import FP8Manager
from .optimizer.fp8_adamw import CoatAdamW

logger = logging.get_logger(__name__)

def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK") or 0)

class CoatFP8Trainer(Trainer):
    def __init__(self, coat_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qargs = coat_args  # Additional parameter

    def create_optimizer(self):
        """
        Setup the FP8 optimizer.
        """
        if get_local_rank() == 0:
            print("Setup the COAT FP8 optimizer")
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_kwargs2 = {
                "lr": self.args.learning_rate,
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
            if self.qargs is not None:
                optimizer_kwargs2["qargs"] = self.qargs
                
            self.optimizer = CoatAdamW(optimizer_grouped_parameters, **optimizer_kwargs2)

        return self.optimizer
    
    # Possibly, put this into trainers's logic
    # if total_batched_samples % args.gradient_accumulation_steps == 0:
    #     FP8Manager.is_first_microbatch = True
    # else:
    #     FP8Manager.is_first_microbatch = False
