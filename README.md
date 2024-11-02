<p align="center" style="border-radius: 10px">
  <img src="docs/figs/COAT.png" width="30%" alt="logo"/>
</p>

# 🚀COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training

Code will be coming soon in early Nov. Please stay tuned!

### [[paper]](https://arxiv.org/abs/2410.19313) | 

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
