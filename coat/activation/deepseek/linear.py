import torch
import deep_gemm

def deepseek_check_stride(x_fp8, y_fp8):
    assert x_fp8[0].stride(1) == 1
    assert x_fp8[1].stride(0) == 1
    assert y_fp8[0].stride(1) == 1
    assert y_fp8[1].stride(1) == 1

def fp8_deepseek_linear_forward(x, s, w, s_w, QB, bias=None):
    x_fp8, w_fp8 = (x, s), (w, s_w)
    out = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=torch.bfloat16)
    
    deepseek_check_stride(x_fp8, w_fp8)
    deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, w_fp8, out)
    return out

def fp8_deepseek_linear_backward(
    x_t, s_t, g, s_g, g_t, s_g_t, w_t, s_w_t
):
    # Dgrad
    g_fp8, w_t_fp8 = (g, s_g), (w_t, s_w_t)
    y = torch.empty((g.shape[0], w_t.shape[0]), device=x_t.device, dtype=torch.bfloat16)
    
    deepseek_check_stride(g_fp8, w_t_fp8)
    deep_gemm.gemm_fp8_fp8_bf16_nt(g_fp8, w_t_fp8, y)

    # Wgrad
    x_t_fp8, g_t_fp8 = (x_t, s_t), (g_t, s_g_t)
    w_g_t = torch.empty((x_t.shape[0], g_t.shape[0]), device=x_t.device, dtype=torch.bfloat16)

    deepseek_check_stride(x_t_fp8, g_t_fp8)
    deep_gemm.gemm_fp8_fp8_bf16_nt(x_t_fp8, g_t_fp8, w_g_t)

    # We need to tranpose the gradient to match the size
    return y, w_g_t.t().contiguous()
    