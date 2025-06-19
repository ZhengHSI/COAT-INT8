# 测试int8矩阵乘法性能
# python tests/test_int8_matmul.py --skip-accuracy --fast

import torch
import time
import triton
import triton.testing
import matplotlib.pyplot as plt
import numpy as np
from coat.activation.real_quantization.linear import int8matmul, fp8matmul
from coat.activation.real_quantization._quantize_pertensor_transpose import fp8_quantize_pertensor_transpose
import argparse

# 在文件开头定义全局变量
global_args = None

class PrecisionSpeedTester:
    def __init__(self, M=1024, K=1024, N=1024, device='cuda'):
        """初始化测试器
        Args:
            M, K, N: 矩阵维度
            device: 运行设备
        """
        self.M, self.K, self.N = M, K, N
        self.device = device
        
        # 生成fp32基准数据
        self.a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
        self.b_fp32 = torch.randn(K, N, device=device, dtype=torch.float32)
        self.bias = torch.randn(N, device=device, dtype=torch.float32)
        
        # 准备各种精度的数据
        self.prepare_all_precisions()
        
    def quantize_tensor(self, x, num_bits=8):
        """通用量化函数"""
        qmax = 2 ** (num_bits - 1) - 1
        scale = x.abs().max() / qmax
        x_int = torch.clamp((x / scale).round(), -qmax, qmax).to(torch.int8 if num_bits == 8 else torch.float32)
        return x_int, scale
        
    def prepare_all_precisions(self):
        """准备所有精度的输入数据"""
        # FP16
        self.a_fp16 = self.a_fp32.half()
        self.b_fp16 = self.b_fp32.half()
        self.bias_fp16 = self.bias.half()
        
        # INT8
        self.a_int8, self.scale_a_int8 = self.quantize_tensor(self.a_fp32, num_bits=8)
        self.b_int8, self.scale_b_int8 = self.quantize_tensor(self.b_fp32, num_bits=8)
        self.scale_c_int8 = 1.0
        
        # FP8
        QB = 16
        fp8type = "E4M3"
        self.a_fp8, self.scale_a_fp8, _ = fp8_quantize_pertensor_transpose(
            self.a_fp32, QB, fp8type, transpose_output_2d=False)
        self.b_fp8, self.scale_b_fp8, _ = fp8_quantize_pertensor_transpose(
            self.b_fp32, QB, fp8type, transpose_output_2d=False)
        self.QB = QB
        
    def benchmark(self, func, repeat=100, warmup=25, cache_key=None):
        """使用triton的benchmark函数"""
        if cache_key and hasattr(self, '_bench_cache') and cache_key in self._bench_cache:
            return self._bench_cache[cache_key]
            
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(
            func, 
            warmup=warmup, 
            rep=repeat,
            quantiles=quantiles
        )
        
        if cache_key:
            if not hasattr(self, '_bench_cache'):
                self._bench_cache = {}
            self._bench_cache[cache_key] = (ms, min_ms, max_ms)
        
        return ms
    
    def benchmark_old(self, func, repeat=100, warmup=25):
        """性能测试函数"""
        # 预热
        for _ in range(warmup):
            func()
            torch.cuda.synchronize()
        
        # 计时
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            func()
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # 转换为毫秒
        
        # 排序后取中间值
        times.sort()
        # 去掉最高和最低的10%
        trim_size = int(repeat * 0.1)
        times = times[trim_size:-trim_size]
        
        median = times[len(times)//2]
        min_time = times[0]
        max_time = times[-1]
        
        return median, min_time, max_time
    
    def compare_results(self, name, ref, test):
        """结果对比函数"""
        diff = (ref - test).abs()
        print(f"\n{name}:")
        print(f"  最大绝对误差: {diff.max().item():.6f}")
        print(f"  平均绝对误差: {diff.mean().item():.6f}")
        print(f"  相对误差: {(diff / (ref.abs() + 1e-6)).mean().item():.6f}")
        print(f"  参考均值: {ref.abs().mean().item():.6f}")
        print(f"  测试均值: {test.abs().mean().item():.6f}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print(f"\n=== 测试矩阵大小: {self.M}x{self.K}x{self.N} ===")
        
        if not global_args.skip_accuracy:
            # 1. 计算基准结果 (FP32)
            c_fp32 = torch.matmul(self.a_fp32, self.b_fp32) + self.bias
            
            # 2. FP16测试
            c_fp16 = torch.matmul(self.a_fp16, self.b_fp16) + self.bias_fp16
            
            # 3. INT8测试
            c_int8 = int8matmul(
                self.a_int8, self.b_int8,
                self.scale_a_int8, self.scale_b_int8,
                self.scale_c_int8, bias=self.bias
            )
            
            # 4. FP8测试
            c_fp8 = fp8matmul(
                self.a_fp8, self.b_fp8, False,
                self.scale_a_fp8, self.scale_b_fp8,
                self.QB, bias=self.bias
            )
            
            # 精度对比（如果不跳过）
            accuracy_results = {
                "FP32": c_fp32,
                "FP16": c_fp16.float(),
                "INT8": c_int8,
                "FP8": c_fp8
            }
        
            print("\n=== 精度对比 ===")
            for name, test in accuracy_results.items():
                self.compare_results(f"{name} vs FP32", c_fp32, test)
        
        # 性能测试
        print("\n=== 性能测试 ===")
        test_funcs = {
            "FP32": lambda: torch.matmul(self.a_fp32, self.b_fp32) + self.bias,
            "FP16": lambda: torch.matmul(self.a_fp16, self.b_fp16) + self.bias_fp16,
            "INT8": lambda: int8matmul(
                self.a_int8, self.b_int8,
                self.scale_a_int8, self.scale_b_int8,
                self.scale_c_int8, bias=self.bias
            ),
            "FP8": lambda: fp8matmul(
                self.a_fp8, self.b_fp8, False,
                self.scale_a_fp8, self.scale_b_fp8,
                self.QB, bias=self.bias
            )
        }
        
        # 运行性能测试
        times = {}
        for name, func in test_funcs.items():
            # Triton benchmark
            ms, min_ms, max_ms = triton.testing.do_bench(
                func,
                warmup=25,
                rep=100,
                quantiles=[0.5, 0.2, 0.8]
            )
            
            times[name] = {
                "median": ms,
                "min": min_ms,
                "max": max_ms,
                "tflops": 2 * self.M * self.N * self.K * 1e-12 / (ms * 1e-3),
                "min_tflops": 2 * self.M * self.N * self.K * 1e-12 / (max_ms * 1e-3),
                "max_tflops": 2 * self.M * self.N * self.K * 1e-12 / (min_ms * 1e-3)
            }
            
            # 如果不跳过老方法，则运行老方法测试
            if not global_args.skip_old_method:
                old_time, old_min, old_max = self.benchmark_old(func)  # 已经是毫秒单位
                times[name].update({
                    "old_time": old_time,
                    "old_min": old_min,
                    "old_max": old_max,
                    "old_tflops": 2 * self.M * self.N * self.K * 1e-12 / (old_time * 1e-3),
                    "old_min_tflops": 2 * self.M * self.N * self.K * 1e-12 / (old_max * 1e-3),
                    "old_max_tflops": 2 * self.M * self.N * self.K * 1e-12 / (old_min * 1e-3)
                })
                print(f"  Old Method:")
                print(f"    中位数耗时: {old_time:.3f} ms")
                print(f"    耗时范围: [{old_min:.3f}, {old_max:.3f}] ms")
                print(f"    TFLOPS: {times[name]['old_tflops']:.2f} [{times[name]['old_min_tflops']:.2f}, {times[name]['old_max_tflops']:.2f}]")
                
            print(f"\n{name}:")
            print(f"  Triton Benchmark:")
            print(f"    中位数耗时: {ms:.3f} ms")
            print(f"    耗时范围: [{min_ms:.3f}, {max_ms:.3f}] ms")
            print(f"    TFLOPS: {times[name]['tflops']:.2f} [{times[name]['min_tflops']:.2f}, {times[name]['max_tflops']:.2f}]")
        
        # 打印加速比
        print(f"\n=== Triton Benchmark 加速比 ===")
        for baseline in ["FP32", "FP16"]:
            print(f"\n相对{baseline}的加速比:")
            baseline_time = times[baseline]["median"]
            for name, time_data in times.items():
                if name != baseline:
                    speedup = baseline_time / time_data["median"]
                    min_speedup = times[baseline]["min"] / time_data["max"]
                    max_speedup = times[baseline]["max"] / time_data["min"]
                    print(f"{name:>6} 加速比: {speedup:.2f}x [{min_speedup:.2f}x, {max_speedup:.2f}x]")
        
        if not global_args.skip_old_method:
            print(f"\n=== Old Method 加速比 ===")
            for baseline in ["FP32", "FP16"]:
                print(f"\n相对{baseline}的加速比:")
                baseline_time = times[baseline]["old_time"]
                for name, time_data in times.items():
                    if name != baseline:
                        speedup = baseline_time / time_data["old_time"]
                        print(f"{name:>6} 加速比: {speedup:.2f}x")
        
        if not global_args.skip_accuracy:
            return {
                "FP32": times["FP32"],
                "FP16": times["FP16"],
                "INT8": times["INT8"],
                "FP8": times["FP8"],
                "accuracy_results": accuracy_results
            }
        else:
            return {
                "FP32": times["FP32"],
                "FP16": times["FP16"],
                "INT8": times["INT8"],
                "FP8": times["FP8"]
            }

def plot_performance_curves(size_results, skip_old_method=False):
    """绘制性能曲线"""
    sizes = list(size_results.keys())
    precisions = ["FP32", "FP16", "INT8", "FP8"]
    
    if skip_old_method:
        # 只创建一行三列的子图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. TFLOPS曲线
        for prec in precisions:
            tflops = [size_results[size][prec]["tflops"] for size in sizes]
            ax1.plot(sizes, tflops, marker='o', label=prec)
        ax1.set_xlabel("Matrix Size")
        ax1.set_ylabel("TFLOPS")
        ax1.set_title("Performance in TFLOPS")
        ax1.grid(True)
        ax1.legend()
        
        # 2. 相对FP32加速比
        for prec in precisions[1:]:  # 跳过FP32
            speedups = [size_results[size]["FP32"]["median"] / size_results[size][prec]["median"] 
                       for size in sizes]
            ax2.plot(sizes, speedups, marker='o', label=prec)
        ax2.set_xlabel("Matrix Size")
        ax2.set_ylabel("Speedup over FP32")
        ax2.set_title("Triton Benchmark: vs FP32")
        ax2.grid(True)
        ax2.legend()
        
        # 3. 相对FP16加速比
        for prec in [p for p in precisions if p != "FP16"]:  # 跳过FP16
            speedups = [size_results[size]["FP16"]["median"] / size_results[size][prec]["median"] 
                       for size in sizes]
            ax3.plot(sizes, speedups, marker='o', label=prec)
        ax3.set_xlabel("Matrix Size")
        ax3.set_ylabel("Speedup over FP16")
        ax3.set_title("Triton Benchmark: vs FP16")
        ax3.grid(True)
        ax3.legend()
    
    else:
        # 创建两行三列的子图
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
        
        # === 第一行：triton benchmark 结果 ===
        # 1. TFLOPS曲线
        for prec in precisions:
            tflops = [size_results[size][prec]["tflops"] for size in sizes]
            ax1.plot(sizes, tflops, marker='o', label=prec)
        ax1.set_xlabel("Matrix Size")
        ax1.set_ylabel("TFLOPS")
        ax1.set_title("Performance in TFLOPS")
        ax1.grid(True)
        ax1.legend()
        
        # 2. 相对FP32加速比
        for prec in precisions[1:]:  # 跳过FP32
            speedups = [size_results[size]["FP32"]["median"] / size_results[size][prec]["median"] 
                       for size in sizes]
            ax2.plot(sizes, speedups, marker='o', label=prec)
        ax2.set_xlabel("Matrix Size")
        ax2.set_ylabel("Speedup over FP32")
        ax2.set_title("Triton Benchmark: vs FP32")
        ax2.grid(True)
        ax2.legend()
        
        # 3. 相对FP16加速比
        for prec in [p for p in precisions if p != "FP16"]:  # 跳过FP16
            speedups = [size_results[size]["FP16"]["median"] / size_results[size][prec]["median"] 
                       for size in sizes]
            ax3.plot(sizes, speedups, marker='o', label=prec)
        ax3.set_xlabel("Matrix Size")
        ax3.set_ylabel("Speedup over FP16")
        ax3.set_title("Triton Benchmark: vs FP16")
        ax3.grid(True)
        ax3.legend()
        
        # === 第二行：老方法结果 ===
        # 4. TFLOPS曲线
        for prec in precisions:
            tflops = [size_results[size][prec]["old_tflops"] for size in sizes]
            ax4.plot(sizes, tflops, marker='o', label=prec)
        ax4.set_xlabel("Matrix Size")
        ax4.set_ylabel("TFLOPS")
        ax4.set_title("Old Method: TFLOPS")
        ax4.grid(True)
        ax4.legend()
        
        # 5. 相对FP32加速比
        for prec in precisions[1:]:  # 跳过FP32
            speedups = [size_results[size]["FP32"]["old_time"] / size_results[size][prec]["old_time"] 
                       for size in sizes]
            ax5.plot(sizes, speedups, marker='o', label=prec)
        ax5.set_xlabel("Matrix Size")
        ax5.set_ylabel("Speedup over FP32")
        ax5.set_title("Old Method: vs FP32")
        ax5.grid(True)
        ax5.legend()
        
        # 6. 相对FP16加速比
        for prec in [p for p in precisions if p != "FP16"]:  # 跳过FP16
            speedups = [size_results[size]["FP16"]["old_time"] / size_results[size][prec]["old_time"] 
                       for size in sizes]
            ax6.plot(sizes, speedups, marker='o', label=prec)
        ax6.set_xlabel("Matrix Size")
        ax6.set_ylabel("Speedup over FP16")
        ax6.set_title("Old Method: vs FP16")
        ax6.grid(True)
        ax6.legend()
    
    plt.tight_layout()
    plt.savefig("matmul_performance_analysis.png")
    plt.close()

def save_summary_results(size_results, skip_old_method=False, skip_accuracy=False, output_file="matmul_summary.txt"):
    """保存汇总结果到文件"""
    sizes = list(size_results.keys())
    precisions = ["FP32", "FP16", "INT8", "FP8"]
    
    with open(output_file, "w") as f:
        # 1. 精度汇总（如果不跳过）
        if not skip_accuracy:
            f.write("="*50 + "\n")
            f.write("精度测试汇总\n")
            f.write("="*50 + "\n\n")
            
            for size in sizes:
                f.write(f"矩阵大小: {size}x{size}x{size}\n")
                f.write("-"*30 + "\n")
                
                # 获取FP32结果作为基准
                c_fp32 = size_results[size]["accuracy_results"]["FP32"]
                
                for prec in ["FP16", "INT8", "FP8"]:
                    c_test = size_results[size]["accuracy_results"][prec]
                    diff = (c_fp32 - c_test).abs()
                    rel_diff = (diff / (c_fp32.abs() + 1e-6)).mean()
                    
                    f.write(f"{prec} vs FP32:\n")
                    f.write(f"  最大绝对误差: {diff.max().item():.6f}\n")
                    f.write(f"  平均绝对误差: {diff.mean().item():.6f}\n")
                    f.write(f"  平均相对误差: {rel_diff.item():.6f}\n")
                f.write("\n")
        
        # 2. 性能汇总
        f.write("\n" + "="*50 + "\n")
        f.write("性能测试汇总\n")
        f.write("="*50 + "\n\n")
        
        # 2.1 TFLOPS汇总
        f.write("TFLOPS 性能:\n")
        f.write("-"*30 + "\n")
        f.write("Size".ljust(10))
        for prec in precisions:
            f.write(f"{prec:>12}")
        f.write("\n")
        
        for size in sizes:
            f.write(f"{size:<10}")
            for prec in precisions:
                tflops = size_results[size][prec]["tflops"]
                f.write(f"{tflops:>12.2f}")
            f.write("\n")
        
        # 2.2 加速比汇总
        if not global_args.skip_old_method:
            methods = ["Triton Benchmark", "Old Method"]
        else:
            methods = ["Triton Benchmark"]
        for method in methods:
            f.write(f"\n{method} 加速比:\n")
            f.write("-"*30 + "\n")
            
            # vs FP32
            f.write("\n相对FP32加速比:\n")
            f.write("Size".ljust(10))
            for prec in precisions[1:]:  # 跳过FP32
                f.write(f"{prec:>12}")
            f.write("\n")
            
            for size in sizes:
                f.write(f"{size:<10}")
                baseline_time = size_results[size]["FP32"]["median" if method == "Triton Benchmark" else "old_time"]
                for prec in precisions[1:]:
                    time_key = "median" if method == "Triton Benchmark" else "old_time"
                    speedup = baseline_time / size_results[size][prec][time_key]
                    f.write(f"{speedup:>12.2f}x")
                f.write("\n")
            
            # vs FP16
            f.write("\n相对FP16加速比:\n")
            f.write("Size".ljust(10))
            for prec in [p for p in precisions if p != "FP16"]:
                f.write(f"{prec:>12}")
            f.write("\n")
            
            for size in sizes:
                f.write(f"{size:<10}")
                baseline_time = size_results[size]["FP16"]["median" if method == "Triton Benchmark" else "old_time"]
                for prec in [p for p in precisions if p != "FP16"]:
                    time_key = "median" if method == "Triton Benchmark" else "old_time"
                    speedup = baseline_time / size_results[size][prec][time_key]
                    f.write(f"{speedup:>12.2f}x")
                f.write("\n")
            

def main():
    global global_args
    
    parser = argparse.ArgumentParser(description='矩阵乘法性能测试')
    parser.add_argument('--enable-old-method', action='store_true', 
                       help='是否启用old method测试（默认关闭）')
    parser.add_argument('--skip-accuracy', action='store_true',
                       help='是否跳过精度对比测试（默认开启）')
    parser.add_argument('--sizes', type=str, default='all',
                       help='要测试的矩阵大小，可以是"small", "medium", "large", "xlarge"或"all"')
    parser.add_argument('--fast', action='store_true', 
                       help='快速测试模式，只测试少量尺寸')
    global_args = parser.parse_args()
    
    # 反转skip_old_method的逻辑
    global_args.skip_old_method = not global_args.enable_old_method
    
    # 根据参数选择测试尺寸
    size_configs = {
        'small': [256, 512, 768],  # 减少小尺寸的测试点
        'medium': [1024, 2048, 3072],  # 选择典型的中等尺寸
        'large': [4096, 6144, 8192],  # 选择典型的大尺寸
        'xlarge': [10240, 12288]  # 只保留两个超大尺寸
    }
    
    if global_args.fast:
        sizes = [1024]  # 只测试三个典型尺寸
    elif global_args.sizes == 'all':
        sizes = (size_configs['small'] + size_configs['medium'] + 
                size_configs['large'] + size_configs['xlarge'])
    else:
        sizes = size_configs.get(global_args.sizes, size_configs['medium'])
    
    # 收集所有结果
    all_results = {}
    for size in sizes:
        print(f"\n{'='*20} 测试矩阵大小: {size}x{size}x{size} {'='*20}")
        tester = PrecisionSpeedTester(size, size, size)
        all_results[size] = tester.run_all_tests()
    
    # 绘制性能曲线
    plot_performance_curves(all_results, skip_old_method=global_args.skip_old_method)
    print("\n性能曲线已保存至 matmul_performance_comparison.png")
    
    # 保存汇总结果
    save_summary_results(all_results, 
                        skip_old_method=global_args.skip_old_method,
                        skip_accuracy=global_args.skip_accuracy)
    print("\n详细结果已保存至 matmul_summary.txt")

if __name__ == "__main__":
    main()