<p align="center" style="border-radius: 10px">
  <img src="docs/figs/COAT.png" width="30%" alt="logo"/>
</p>

# 🚀COAT: Compressing Optimizer States and Activation for Memory-Efficient FP8 Training

### [[paper]](https://arxiv.org/abs/2410.19313) [[website]](https://nvlabs.github.io/COAT/)

<p align="center" style="border-radius: 10px">
  <img src="docs/figs/FP8PrecisionFlow.png" width="90%" alt="logo"/>
</p>

## 💡 Introduction
We introduce COAT (Compressing Optimizer states and Activation for Memory-Efficient FP8 Training), a novel method designed to optimize the memory efficiency of training large models by compressing optimizer states and activations using FP8 quantization.

This technique allows:

- Reduced end-to-end memory footprint by 1.54× 
- Speedup training by 1.43× while maintaining model accuracy
- Double the batch size and utilize GPU better
- Scalable performance for large-scale AI models

By leveraging FP8 precision, COAT significantly decreases memory usage by 1.54×, which is critical for training large models on limited hardware resources.

## 🔥 News

- [2024/10] [[paper]](https://arxiv.org/abs/2410.19313) is on arxiv!

## 📚 Abstract
FP8 training has emerged as a promising method for improving training efficiency. Existing frameworks accelerate training by applying FP8 computation to linear layers while leaving optimizer states and activations in higher precision, which fails to fully optimize memory usage. This paper introduces COAT (**C**ompressing **O**ptimizer States and **A**ctivations for FP8 **T**raining), a novel FP8 training framework designed to significantly reduce memory footprint when training large models. 

COAT addresses current limitations through two key innovations: (1) **Dynamic Range Expansion**, which aligns optimizer state distributions more closely with the FP8 representation range, thereby reducing quantization error, and (2) **Mixed-Granularity Activation Quantization**, which optimizes activation memory using a combination of per-tensor and per-group quantization strategies. 

Experiments demonstrate that COAT effectively reduces end-to-end training memory footprint by **1.54×** compared to BF16 while achieving nearly lossless performance across various tasks, such as Large Language Model pretraining and fine-tuning and Vision Language Model training. COAT also achieves a **1.43×** end-to-end training speedup compared to BF16, performing on par with or surpassing TransformerEngine's speedup. COAT enables efficient full-parameter training of large models on fewer GPUs, and facilitates doubling the batch size in distributed training settings, providing a practical solution for scaling large-scale model training.

## ⚙️ Installation
```bash
git clone --recurse-submodules https://github.com/NVlabs/COAT.git
cd COAT

# Create the conda environment and install coat
chmod +x environment_setup.sh
./environment_setup.sh coat

conda activate coat

# Install OLMo
cd examples/OLMo/
pip install -e .[all]
```

## 📊 Memory Saving, Speedup, and Accuracy
### Memory Saving and Speedup
In all multi-GPU training setting, COAT can double the micro-batch size and therefore lead to even higher speedup. For example, our method can achieve $2.25\times$ speedup when training Llama-2-13B on 4-GPUs since we can effectively increase the batch size to 2.

Overall, COAT significantly reduces end-to-end memory usage by up to $1.55\times$ and speeds up the end-to-end training by nearly $1.44\times$. This facilitates full-parameter training on fewer GPUs, which is particularly beneficial for larger language models.

<p align="center" style="border-radius: 10px">
  <img src="docs/figs/end_to_end_bench.png" width="90%" alt="logo"/>
</p>

### Accuracy Experiments
#### OLMo-Pretraining
We pretrained the [OLMo-1B and OLMo-7B](https://github.com/allenai/OLMo.git) models on Dolma, as outlined in the official report. The training curves and downstream task performance aligned closely with the results from BF16 training and the TransformerEngine baseline, confirming the effectiveness of COAT.

<p align="center" style="border-radius: 10px">
  <img src="docs/figs/OLMo-1B7BTraining.png" width="90%" alt="logo"/>
</p>

#### Downstream Task Performance
We validated the effectiveness of our method using real-world examples. In the Image Captioning task, the [VILA model](https://github.com/NVlabs/VILA) trained with COAT demonstrated the ability to accurately summarize and identify key elements in the figures, perform on-par with models trained with BF16.

<p align="center" style="border-radius: 10px">
  <img src="docs/figs/VILAExample.png" width="90%" alt="logo"/>
</p>

<style>
h3 {
    text-align: center;
}
</style>

## 🔍 Key Observations
<div align="center">
<h3> Part1: FP8 Optimizer </h3>
</div>

#### Difficulty of FP8 quantization for optimizer states
We find that current quantization methods can not fully utilize the representation range of FP8 and therefore lead to a large quantization error when quantizing optimizer states with per-group quantization. For the <a href="https://arxiv.org/abs/2209.05433" target="_blank">E4M3 format</a>, we hope the dynamic range of the quantization group X should cover the entire span between the minimum representable value of E4M3 (0.00195) and the maximum representable value of E4M3 (448) to fully utilize its representation ability. However, the dynamic range of E4M3 is usually **under-utilized**: The dynamic range of E4M3 is about 2e5, but the dynamic range of first order momentum is usually 1e3, and the dynamic range of second order momentum is usually 1e1. This make the quantization error really large.

<p align="center" style="border-radius: 10px">
  <img src="docs/figs/rk_11_E4M3_E4M3_final.png" width="90%" alt="logo"/>
</p>

#### Our solution: Dynamic Range Expansion:
We introduce a expand function f(·) before quantization to expand the dynamic range of the quantization group and align it with E4M3. The expand function we use is:

![f(x) = \operatorname{sign}(x) \cdot |x|^k](https://latex.codecogs.com/svg.image?f(x)=sign(x)x^k)

where k is a parameter we calculate on-the-fly. When k > 1 , the dynamic range will be enlarged and become closer to the dynamic range of E4M3. The optimal k can be directly calculated and can fully utilize the representation range of E4M3, while the original quantization method can only utilize a small portion of it. Our dynamic range expansion method can greatly reduce the quantization error and fully utilize the dynamic range of E4M3. We find that E4M3 is more suitable for first-order momentum than E5M2. For second-order momentum, although E4M3 is better than E5M2 in the original setting, their quantization error is nearly the same after applying our expand function. Therefore, we propose to use the E4M3 + E4M3 quantization strategy or E4M3 + E5M2 quantization strategy when quantizing the optimizer states.

<p align="center" style="border-radius: 10px">
  <img src="docs/figs/RangeExpansion.png" width="90%" alt="logo"/>
</p>

<div align="center">
<h3> Part2: FP8 Activation </h3>
</div>

#### Motivation: Non-linear layers costs large memory footprint
In the forward pass of neural networks, activations must be preserved for the backward pass to calculate gradients. Non-linear layers typically account for approximately 50% of the memory footprint in the Llama model series. In contrast, linear layers contribute less than 25%. Therefore, it is essential to optimize both linear and non-linear layers to reduce activation memory footprint.

<p align="center" style="border-radius: 10px">
  <img src="docs/figs/NonlinearLinear.png" width="90%" alt="logo"/>
</p>

<p align="center" style="border-radius: 10px">
  <img src="docs/figs/DE8QuantError.png" width="75%" alt="logo"/>
</p>

#### Our Solution: Mixed Granularity FP8 Precision Flow
FP8 precision flow requires the input and output of all linear and non-linear layers in FP8. By directly saving the input tensor in FP8 format for the backward pass, we eliminate the need for an extra quantization operation, which reduces the associated overhead. FP8 precision flow natually reduce the memory footprint for non-linears and linear layers by 50%, since they only need to save FP8 activations, not BF16. To further improve the accurateness of this method, we propose to vary the quantization granularity across different layers to balance precision and efficiency in a mixed-granularity manner. For non-linear layers, VS-Quant or PerBlock Quant methods are well-suited due to their fine-grained and precise nature. For linear layers, we apply per-tensor quantization to maximize the performance of Tensor Cores. We observe that quantizing the input of layernorm across multiple token axes is detrimental to accuracy, and therefore decide to apply per-group quantization to non-linear layers.

#### Group Scaling: Efficient Just-in-time Scaling
To perform per-tensor quantization, the maximum absolute value of the tensor needs to be calculated through max reduction, adding a lot of overhead. In our Group Scaling, we address these problems by splitting the max reduction into two stages: (1) performing max reduction on each 1 × G element and storing the results as intermediate values; (2) applying max reduction on the intermediate tensor to obtain the per-tensor max value. The first stage can be seamlessly fused with the previous operation, adding minimal overhead, while the second stage is more efficient than doing max reduction on the entire tensor, as the intermediate result is G× smaller than the original tensor.

<p align="center" style="border-radius: 10px">
  <img src="docs/figs/mixedgranularity.png" width="90%" alt="logo"/>
</p>


## 📖 Examples - Llama-2 model training

### Use COAT's FP8 Activation and FP8 Optimizer
We have support Llama 2/3 to use COAT FP8 activation and FP8 optimizer feature.
Just need several lines of code to support COAT.

To use the FP8 activation feature, please first convert your model into COAT's format:
```bash
export MODEL_NAME="meta-llama/Llama-2-7b-hf"
export CONVERTED_MODEL_PATH="converted_models/llama-2-7b"

export COAT_PATH=$(pip show coat | grep "Editable project location" | awk -F': ' '{print $2}')
echo "COAT package is located at: $COAT_PATH"

python $COAT_PATH/coat/activation/models/coat_llama_convert_from_hf.py \
    --model_name $MODEL_NAME \
    --save_path $CONVERTED_MODEL_PATH \
    --quantize_model true \
    --fabit E4M3 \
    --fwbit E4M3 \
    --fobit E4M3 \
    --bwbit E5M2 \
    --babit E5M2 \
    --bobit E5M2 \
    --group_size 16
```

To use the FP8 optimizer feature, add these arguments to your training script:
```bash
    --first_order_expansion true \
    --second_order_expansion true \
    --first_order_bit E4M3 \
    --second_order_bit E4M3 \
    --qgroup_size 128 \
    --expand_min 16
```

Then add these lines to enable FP8 Activation and FP8 Optimizer in the training process
```python
# import coat
from coat.activation.models._fp8_quantization_config import QuantizationConfig
from coat.fp8_trainer import CoatFP8Trainer

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    # fp8_model_name_or_path is the CONVERTED_MODEL_PATH you create in the section above
    fp8_model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

def train():
    # ----- some irrelevant code -----
    # Add the quantization argument for FP8 optimizer states
    parser = transformers.HfArgumentParser( # NOTE: FP8
        (ModelArguments, DataArguments, TrainingArguments, QuantizationConfig)
    )

    # Load the converted checkpoint to support COAT for FP8 activation
    fp8_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.fp8_model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map=device_map
    )
    fp8_model.config.use_cache = False

    # Replace the trainer with ours trainer
    trainer = CoatFP8Trainer(
        model=fp8_model, tokenizer=tokenizer, args=training_args, coat_args=quantization_args, **data_module
    )
```

An example for the process above is to run the following code to start a real FP8 training example on [ToolBench](https://github.com/OpenBMB/ToolBench)!
```bash
cd examples/ToolBench
# Download the data ToolBench requires
gdown --id 1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk --output data.zip
# Unzip the data
python preprocess/unzip_finetune_data.py

# Run COAT FP8 Training on Llama-2-7B, should achieve 2.80s/it on 8 * H100, which is 26% speedup compared with BF16 training.
bash scripts/train_toolllama_fp8.sh

# Run BF16 baseline on Llama-2-7B, should achieve 3.53s/it on 8 * H100.
bash scripts/train_toolllama_bf16.sh
```


## 📖 Examples: OLMo Pretraining
### Dataset
First Prepare the training data and validation data following the instructions in OLMo's readme.
```bash
# An example to download the Dolma dataset
wget https://olmo-data.org/preprocessed/olmo-mix/v1_5/gpt-neox-20b-pii-special/part-000-00000.npy
```

### Training script for OLMo 1B
```bash
# Reproduce pretraining of OLMo-1B
cd examples/OLMo/

# BF16
torchrun --nproc_per_node=8 scripts/train.py configs/reproduce/OLMo-1B-reproduce.yaml

# COAT
torchrun --nproc_per_node=8 scripts/train.py configs/coat/OLMo-1B-COAT-BOTH.yaml

# Only linear layer in FP8
torchrun --nproc_per_node=8 scripts/train.py configs/fp8linear/OLMo-1B-COAT-FP8Linear.yaml
```

### Training script for OLMo 7B
```bash
# Reproduce pretraining of OLMo-7B
cd examples/OLMo/

# BF16
torchrun --nproc_per_node=8 scripts/train.py configs/reproduce/OLMo-7B-reproduce.yaml

# COAT
torchrun --nproc_per_node=8 scripts/train.py configs/coat/OLMo-7B-COAT-BOTH.yaml
```


## 🔁 Benchmark speedup and memory saving

### Reproduce 1: End-to-End Memory Reduction and Speedup on 7B Model
```bash
# Reproduce the memory reduction in Table 7 of our paper.
cd examples/OLMo/

# BF16
MEMORY_BENCH=1 torchrun --nproc_per_node=4 scripts/train.py configs/reproduce/OLMo-7B-reproduce-MemBench.yaml

# COAT
MEMORY_BENCH=1 torchrun --nproc_per_node=4 scripts/train.py configs/coat/OLMo-7B-COAT-Both-MemBench.yaml
```

```bash
# Reproduce the speedup in Table 7 of our paper.
cd examples/OLMo/

# BF16
SPEED_BENCH=1 torchrun --nproc_per_node=4 scripts/train.py configs/reproduce/OLMo-7B-reproduce-SpeedBench.yaml

# COAT
SPEED_BENCH=1 torchrun --nproc_per_node=4 scripts/train.py configs/coat/OLMo-7B-COAT-Both-SpeedBench.yaml
```

### Reproduce 2: Per TransformerLayer Memory Reduction and Speedup
```bash
# Automatically compare BF16 and COAT
python benchmark/benchmark_olmolayer.py
```


## 💪To-Do List
We will try our best to release

- \[ \] COAT on [TorchTitan](https://github.com/pytorch/torchtitan)
- \[ \] COAT on [FSDP2](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)
- \[ \] COAT on [VILA](https://github.com/NVlabs/VILA)

## FAQ
If you have some problem when installing `qoptim_cuda`, you can try to install cudatoolkit following [this link](https://stackoverflow.com/questions/39379792/install-cuda-without-root)

## Citation
```
@misc{xi2024coatcompressingoptimizerstates,
      title={COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training}, 
      author={Haocheng Xi and Han Cai and Ligeng Zhu and Yao Lu and Kurt Keutzer and Jianfei Chen and Song Han},
      year={2024},
      eprint={2410.19313},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.19313}, 
}
```
