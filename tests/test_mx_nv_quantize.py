import torch
from transformers import AutoModelForCausalLM
from coat.activation.fake_quantization.quantize_function import extract_bit, block_cut, block_quant, block_reshape

# Fix the random seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def eval_quant(input, precision, block_size=-1, symm=True, stochastic=False, epsilon=1e-12, apply_quantize=True, layer_name="test"):
    """
    Evaluate quantization performance for different precision formats and block sizes.
    
    Args:
        input: Input tensor to quantize
        precision: Quantization precision format (e.g., "E4M3", "MXE4M3", "NVE4M3")
        block_size: Block size for group quantization. Use -1 for per-tensor quantization.
        symm: Whether to use symmetric quantization
        stochastic: Whether to use stochastic rounding
        epsilon: Small value to avoid division by zero
        apply_quantize: Whether to apply quantization
        layer_name: Name of the layer for debugging
        
    Returns:
        tuple: (Quantized tensor, Quantized values, Scale factors, MSE)
    """
    # Cut the input into blocks
    Binput = block_cut(input, 1 if block_size > 0 else -1, block_size)
    
    # Apply quantization
    RQInput, q_values, scale_factors = block_quant(
        Binput, symm, precision, stochastic, epsilon, apply_quantize, layer_name
    )
    
    # Reshape back to original shape
    RQInput = block_reshape(RQInput, input, 1 if block_size > 0 else -1, block_size)
    
    # Calculate mean squared error
    mse = (input - RQInput).pow(2).mean().item()
    
    # Print results
    quant_type = "Per-tensor" if block_size <= 0 else f"Group={block_size}"
    print(f'MSE for {precision} {quant_type}: {mse}')
    
    return RQInput, q_values, scale_factors, mse

if __name__ == "__main__":
    input = torch.randn(4096, 4096).cuda()

    # Simulate outlier
    input[:, 100] *= 500
    input[100, :] *= 500

    # Test parameters
    stochastic = False
    epsilon = 1e-12
    apply_quantize = True
    layer_name = "test"
    symm = True
    
    print("\n=== Testing Per-tensor Quantization of E4M3 ===")
    precision, block_size = "E4M3", -1
    
    # Use the evaluate_quantization function
    RQInput_E4M3_tensor, q_e4m3, scale_e4m3, mse_E4M3_tensor = eval_quant(
        input, precision, block_size, symm, stochastic, epsilon, apply_quantize, layer_name
    )
    
    print("\n=== Testing Group = 128 Quantization of E4M3 ===")
    precision, block_size = "E4M3", 128
    RQInput_E4M3_group128, q_e4m3, scale_e4m3, mse_E4M3_group128 = eval_quant(
        input, precision, block_size, symm, stochastic, epsilon, apply_quantize, layer_name
    )
    
    print("\n=== Testing Group = 32 Quantization of MXE5M1 ===")
    precision, block_size = "MXE5M1", 32
    RQInput_MXE5M1_group32, q_mx_e5m1, scale_mx_e5m1, mse_MXE5M1_group32 = eval_quant(
        input, precision, block_size, symm, stochastic, epsilon, apply_quantize, layer_name
    )
    
    print("\n=== Testing Group = 32 Quantization of MXE5M1_plus ===")
    precision, block_size = "MXE5M1_plus", 32
    RQInput_MXE5M1_plus_group32, q_mx_e5m1_plus, scale_mx_e5m1_plus, mse_MXE5M1_plus_group32 = eval_quant(
        input, precision, block_size, symm, stochastic, epsilon, apply_quantize, layer_name
    )
    
    print("\n=== Testing Group = 32 Quantization of MXE4M3 ===")
    precision, block_size = "MXE4M3", 32
    RQInput_MXE4M3_group32, q_mx_e4m3, scale_mx_e4m3, mse_MXE4M3_group32 = eval_quant(
        input, precision, block_size, symm, stochastic, epsilon, apply_quantize, layer_name
    )
    
    print("\n=== Testing Group = 32 Quantization of MXE3M2 ===")
    precision, block_size = "MXE3M2", 32
    RQInput_MXE3M2_group32, q_mx_e3m2, scale_mx_e3m2, mse_MXE3M2_group32 = eval_quant(
        input, precision, block_size, symm, stochastic, epsilon, apply_quantize, layer_name
    )

    print("\n=== Testing Group = 32 Quantization of MXE2M3 ===")
    precision, block_size = "MXE2M3", 32
    RQInput_MXE2M3_group32, q_mx_e2m3, scale_mx_e2m3, mse_MXE2M3_group32 = eval_quant(
        input, precision, block_size, symm, stochastic, epsilon, apply_quantize, layer_name
    )

    print("\n=== Testing Group = 32 Quantization of MXE2M1 ===")
    precision, block_size = "MXE2M1", 32
    RQInput_MXE2M1_group32, q_mx_e2m1, scale_mx_e2m1, mse_MXE2M1_group32 = eval_quant(
        input, precision, block_size, symm, stochastic, epsilon, apply_quantize, layer_name
    )

    print("\n=== Testing Group = 32 Quantization of NVE5M1 ===")
    precision, block_size = "NVE5M1", 32
    RQInput_NVE5M1_group32, q_nv_e5m1, scale_nv_e5m1, mse_NVE5M1_group32 = eval_quant(
        input, precision, block_size, symm, stochastic, epsilon, apply_quantize, layer_name
    )

    print("\n=== Testing Group = 32 Quantization of NVE5M1_plus ===")
    precision, block_size = "NVE5M1_plus", 32
    RQInput_NVE5M1_plus_group32, q_nv_e5m1_plus, scale_nv_e5m1_plus, mse_NVE5M1_plus_group32 = eval_quant(
        input, precision, block_size, symm, stochastic, epsilon, apply_quantize, layer_name
    )

    print("\n=== Testing Group = 16 Quantization of NVE2M1 ===")
    precision, block_size = "NVE2M1", 16
    RQInput_NVE2M1_group16, q_nv_e2m1, scale_nv_e2m1, mse_NVE2M1_group16 = eval_quant(
        input, precision, block_size, symm, stochastic, epsilon, apply_quantize, layer_name
    )

    print("\n=== Testing Group = 32 Quantization of NVE2M1_plus ===")
    precision, block_size = "NVE2M1_plus", 32
    RQInput_NVE2M1_plus_group32, q_nv_e2m1_plus, scale_nv_e2m1_plus, mse_NVE2M1_plus_group32 = eval_quant(
        input, precision, block_size, symm, stochastic, epsilon, apply_quantize, layer_name
    )
