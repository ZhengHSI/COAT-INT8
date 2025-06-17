import torch
from coat.activation.real_quantization.linear import int8matmul, fp8matmul
from coat.activation.real_quantization._quantize_pertensor_transpose import fp8_quantize_pertensor_transpose

def quantize_tensor(x, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    scale = x.abs().max() / qmax
    x_int = torch.clamp((x / scale).round(), -qmax, qmax).to(torch.int8 if num_bits == 8 else torch.float32)
    return x_int, scale

# 1. 随机生成输入
M, K, N = 1024, 1024, 1024
a_fp32 = torch.randn(M, K, device='cuda', dtype=torch.float32)
b_fp32 = torch.randn(K, N, device='cuda', dtype=torch.float32)
bias = torch.randn(N, device='cuda', dtype=torch.float32)

# 2. float32 baseline
c_fp32 = torch.matmul(a_fp32, b_fp32) + bias

# 3. int8量化
a_int8, scale_a = quantize_tensor(a_fp32, num_bits=8)
b_int8, scale_b = quantize_tensor(b_fp32, num_bits=8)
scale_c = 1.0  # 反量化时可设为1.0
c_int8 = int8matmul(a_int8, b_int8, scale_a, scale_b, scale_c, bias=bias)

# 4. fp8量化（用官方API）
QB = 16
fp8type = "E4M3"
a_fp8, scale_a_fp8, _ = fp8_quantize_pertensor_transpose(a_fp32, QB, fp8type, transpose_output_2d=False)
b_fp8, scale_b_fp8, _ = fp8_quantize_pertensor_transpose(b_fp32, QB, fp8type, transpose_output_2d=False)
c_fp8 = fp8matmul(a_fp8, b_fp8, False, scale_a_fp8, scale_b_fp8, QB, bias=bias)

# 5. 误差对比
def compare(name, ref, test):
    diff = (ref - test).abs()
    print(f"{name}:")
    print(f"  max abs diff: {diff.max().item():.6f}")
    print(f"  mean abs diff: {diff.mean().item():.6f}")
    print(f"  ref mean: {ref.abs().mean().item():.6f}, test mean: {test.abs().mean().item():.6f}")

compare("int8 vs float32", c_fp32, c_int8)
compare("fp8 vs float32", c_fp32, c_fp8)
compare("int8 vs fp8", c_fp8, c_int8)

# === 性能测试 ===
import time

def benchmark(func, repeat=10, warmup=3):
    for _ in range(warmup):
        func()
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        func()
        torch.cuda.synchronize()
    end = time.perf_counter()
    avg_time = (end - start) / repeat
    return avg_time

int8_func = lambda: int8matmul(a_int8, b_int8, scale_a, scale_b, scale_c, bias=bias)
fp8_func = lambda: fp8matmul(a_fp8, b_fp8, False, scale_a_fp8, scale_b_fp8, QB, bias=bias)
fp32_func = lambda: torch.matmul(a_fp32, b_fp32) + bias

print("\n=== 性能测试 ===")
print(f"int8matmul 平均耗时: {benchmark(int8_func)*1000:.3f} ms")
print(f"fp8matmul 平均耗时: {benchmark(fp8_func)*1000:.3f} ms")
print(f"float32 matmul 平均耗时: {benchmark(fp32_func)*1000:.3f} ms")

    