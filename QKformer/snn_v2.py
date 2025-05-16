import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial
from timm.models import create_model

__all__ = ['QKFormer', 'SpikingTransformer', 'PPTransformer', 'LiDiffTransformer', 'QKFormer_10_384', 'QKFormer_10_768', 'TokenSpikingTransformer',
            'PushPullTransformer']

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)
        self.mlp1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)
        self.mlp2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x, spike_rate=False):
        T, B, C, H, W = x.shape

        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)
        x_first_spikes = self.mlp1_lif(x)


        x = self.mlp2_conv(x_first_spikes.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        x_second_spikes = self.mlp2_lif(x)

        if spike_rate:
            return x_second_spikes, {
            'mlp1': x_first_spikes.detach(),
            'mlp2': x_second_spikes.detach(),
            }
        else:
            return x_second_spikes, None


class Token_QK_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x, spike_rate=False):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_spikes = self.q_lif(q_conv_out)
        q = q_spikes.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_spikes = self.k_lif(k_conv_out)
        k = k_spikes.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        q = torch.sum(q, dim=3, keepdim=True)
        attn_spikes = self.attn_lif(q)
        x_one_spikes = torch.mul(attn_spikes, k)

        x = x_one_spikes.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x_final_spikes = self.proj_lif(x)

        if spike_rate:
            return x_final_spikes, {
            'q': q_spikes.detach(),
            'k': k_spikes.detach(),
            'attn': attn_spikes.detach(),
            'x_one': x_one_spikes.detach(),
            'x_final': x_final_spikes.detach(),
            }
        else:
            return x_final_spikes, None


class PushPull_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.qe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.qi_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        # self.merge_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x, spike_rate=False):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)

        qe_spikes = self.qe_lif(q_conv_out)
        qi_spikes = self.qi_lif(-q_conv_out)

        qe = qe_spikes.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)
        qi = qi_spikes.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_spikes = self.k_lif(k_conv_out)
        k = k_spikes.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        qe = torch.sum(qe, dim=3, keepdim=True)
        qi = torch.sum(qi, dim=3, keepdim=True)

        attn_spikes = self.attn_lif(qe-qi)
        x_one_spikes = torch.mul(attn_spikes, k)

        x = x_one_spikes.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x_final_spikes = self.proj_lif(x)

        if spike_rate:
            return x_final_spikes, {
            'qe': qe_spikes.detach(),
            'qi': qi_spikes.detach(),
            'k': k_spikes.detach(),
            'attn': attn_spikes.detach(),
            'x_one': x_one_spikes.detach(),
            'x_final': x_final_spikes.detach(),
            }
        else:
            return x_final_spikes, None

class PP_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        assert (dim // 2) % num_heads == 0, f"dim//2 {dim//2} must be divisible by num_heads {num_heads} for qe/qi split."
        self.scale = 0.125
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_head_dim = (dim // 2) // num_heads # Head dimension for qe and qi

        # --- Query Pathway (Modified for FF-LiDiff) ---
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias) # Output C channels
        self.q_bn = nn.BatchNorm1d(dim)
        # Removed original self.q_lif
        # Added LIFs for excitatory and inhibitory paths
        self.qe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.qi_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.merge_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy') # For the difference

        self.push_w = nn.Parameter(torch.tensor(1.0))
        self.pull_w = nn.Parameter(torch.tensor(1.0))
        self.softplus = nn.Softplus()
        self.pp_bias = nn.Parameter(torch.tensor(0.0))

        # --- Key Pathway (Mostly Unchanged) ---
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')


        # --- Projection Pathway (Unchanged) ---
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        x_flat = x.flatten(3) # Shape: [T, B, C, N]
        x_for_qkv = x_flat.flatten(0, 1) # Shape: [T*B, C, N]

        # --- Query Processing with FF-LiDiff ---
        q_conv_out = self.q_conv(x_for_qkv) # Shape: [T*B, C, N]
        q_bn_out = self.q_bn(q_conv_out).reshape(T, B, C, N)   # Shape: [T*B, C, N]

        qe = self.qe_lif(q_bn_out)
        qi = self.qi_lif(-q_bn_out)

        qe = qe.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,4).contiguous()
        qi = qi.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,4).contiguous()

        pw = self.softplus(self.push_w)
        pl = self.softplus(self.pull_w)

        q_pp = self.merge_lif(pw * qe - pl * qi + self.pp_bias)

        del q_conv_out, q_bn_out, qe, qi, pw, pl


        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2, -1) @ v
        x = (q_pp @ x) * self.scale

        del q_pp, v

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, W, H)

        return x

class LiDiff_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        assert (dim // 2) % num_heads == 0, f"dim//2 {dim//2} must be divisible by num_heads {num_heads} for qe/qi split."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_head_dim = (dim // 2) // num_heads # Head dimension for qe and qi

        # --- Query Pathway (Modified for FF-LiDiff) ---
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias) # Output C channels
        self.q_bn = nn.BatchNorm1d(dim)
        # Removed original self.q_lif
        # Added LIFs for excitatory and inhibitory paths
        self.qe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.qi_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.merge_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy') # For the difference

        # --- Key Pathway (Mostly Unchanged) ---
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        # --- Attention LIF (Moved) ---
        # Original attn_lif applied to summed q, now applied after q*k product
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        # --- Projection Pathway (Unchanged) ---
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x, spike_rate=False):
        T, B, C, H, W = x.shape
        N = H * W

        x = x.flatten(3) # Shape: [T, B, C, N]
        x = x.flatten(0, 1) # Shape: [T*B, C, N]

        # --- Query Processing with FF-LiDiff ---
        q_conv_out = self.q_conv(x) # Shape: [T*B, C, N]
        q_bn_out = self.q_bn(q_conv_out)   # Shape: [T*B, C, N]

        # Split into excitatory (qe) and inhibitory (qi) pathways
        qe, qi = q_bn_out.chunk(2, dim=1) # Shape: [T*B, C/2, N] each

        # Reshape back to [T, B, C/2, N]
        qe = qe.reshape(T, B, C // 2, N)
        qi = qi.reshape(T, B, C // 2, N)

        # Apply separate LIFs to qe and qi
        qe_spikes = self.qe_lif(qe)
        qi_spikes = self.qi_lif(qi)

        # Reshape for multi-head attention
        # Shape: [T, B, num_heads, q_head_dim, N]
        qe = qe_spikes.reshape(T, B, self.num_heads, self.q_head_dim, N)
        qi = qi_spikes.reshape(T, B, self.num_heads, self.q_head_dim, N)

        # --- Attention Calculation with Inhibition ---
        # Sum qe and qi over the head dimension (dim=3)
        qe_sum = torch.sum(qe, dim=3, keepdim=True) # Shape: [T, B, num_heads, 1, N]
        qi_sum = torch.sum(qi, dim=3, keepdim=True) # Shape: [T, B, num_heads, 1, N]

        # Calculate difference and apply merge_lif
        # Note: merge_lif now acts as the attention mechanism driver
        attn_spikes = self.merge_lif(qe_sum - qi_sum) # Shape: [T, B, num_heads, 1, N]

        del q_conv_out, q_bn_out, qe, qi, qe_sum, qi_sum

        # --- Key Processing ---
        k_conv_out = self.k_conv(x) # Shape: [T*B, C, N]
        k_bn_out = self.k_bn(k_conv_out)   # Shape: [T*B, C, N]
        k_bn_out = k_bn_out.reshape(T, B, C, N) # Shape: [T, B, C, N]
        k_spikes = self.k_lif(k_bn_out) # Apply LIF. Shape: [T, B, C, N]

        # Reshape for multi-head attention
        # Shape: [T, B, num_heads, head_dim, N]
        k = k_spikes.reshape(T, B, self.num_heads, self.head_dim, N)


        # Multiply filtered_q (attention driver) with k (value)
        # Broadcasting filtered_q over head_dim of k
        x_one_spikes = torch.mul(attn_spikes, k) # Shape: [T, B, num_heads, head_dim, N]

        del k_conv_out, k_bn_out, k

        # Flatten heads: [T, B, C, N]
        x = x_one_spikes.reshape(T, B, C, N)

        # Apply projection layers
        x = x.flatten(0, 1) # Shape: [T*B, C, N]
        proj_out = self.proj_conv(x) # Shape: [T*B, C, N]
        proj_bn_out = self.proj_bn(proj_out)   # Shape: [T*B, C, N]

        # Reshape back to original image tensor format [T, B, C, H, W]
        proj_bn_out = proj_bn_out.reshape(T, B, C, N)
        final_x = proj_bn_out.reshape(T, B, C, H, W)

        # Apply final LIF
        x_final_spikes = self.proj_lif(final_x)

        if spike_rate:
            return x_final_spikes, {
            'qe': qe_spikes.detach(),
            'qi': qi_spikes.detach(),
            'k': k_spikes.detach(),
            'attn': attn_spikes.detach(),
            'x_one': x_one_spikes.detach(),
            'x_final': x_final_spikes.detach(),
            }
        else:
            return x_final_spikes, None


class Spiking_Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.qkv_mp = nn.MaxPool1d(4)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4).contiguous()

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, W, H)

        return x

class TokenSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, spike_rate=False):
        super().__init__()
        self.tssa = Token_QK_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, spike_rate=False):
        y, attn_spikes = self.tssa(x, spike_rate=spike_rate)
        x = x + y 
        y, mlp_spikes = self.mlp(x, spike_rate=spike_rate)
        x = x + y
        return x, attn_spikes, mlp_spikes

class LiDiffTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, spike_rate=False):
        super().__init__()
        self.tssa = LiDiff_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)


    def forward(self, x, spike_rate=False):
        y, attn_spikes = self.tssa(x, spike_rate=spike_rate)
        x = x + y 
        y, mlp_spikes = self.mlp(x, spike_rate=spike_rate)
        x = x + y
        return x, attn_spikes, mlp_spikes

class PPTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.tssa = PP_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.tssa(x)
        x = x + self.mlp(x)
        return x, None, None

class PushPullTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.tssa = PushPull_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    
    def forward(self, x, spike_rate=False):
        y, attn_spikes = self.tssa(x, spike_rate=spike_rate)
        x = x + y 
        y, mlp_spikes = self.mlp(x, spike_rate=spike_rate)
        x = x + y
        return x, attn_spikes, mlp_spikes


class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.ssa = Spiking_Self_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.ssa(x)
        y, _ = self.mlp(x)
        x = x + y

        return x, None, None


class MS_DownSampling(nn.Module):
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
    ):
        super().__init__()

        # Only the first layer uses MAC operations, subsequent layers use binary spikes
        # which can be implemented as conditional additions on neuromorphic hardware
        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )

    def forward(self, x):
        T, B, _, _, _ = x.shape

        if hasattr(self, "encode_lif"):
            x = self.encode_lif(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()

        return x


class PatchEmbedInit(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj1_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 1)
        self.proj1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_res_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W)
        x = self.proj_lif(x).flatten(0, 1)

        x_feat = x
        x = self.proj1_conv(x)
        x = self.proj1_bn(x).reshape(T, B, -1, H, W)
        x = self.proj1_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H, W).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat  # shortcut

        return x


class PatchEmbeddingStage(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj3_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_res_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, H, W = x.shape
        # Downsampling + Res

        x = x.flatten(0, 1).contiguous()
        x_feat = x

        x = self.proj3_conv(x)
        x = self.proj3_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj3_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x)
        x = self.proj4_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat  # shortcut

        return x


class spiking_transformer(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[8, 8, 8], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T=4, pretrained_cfg=None, transformer_classes = None, spike_rate=False
                 ):
        super().__init__()
        self.spike_rate = spike_rate
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        num_heads = [8, 8, 8]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        if transformer_classes is None:
            raise ValueError("You must pass a list of transformer classes.")
        stage1_class = transformer_classes[0]
        stage2_class = transformer_classes[1]
        stage3_class = transformer_classes[2]

        # Assuming embed_dims is an integer representing the final stage embedding dim (e.g., 256, 384, 768)
        # Channel dimensions for each stage
        c1_embed_dim = embed_dims // 4
        c2_embed_dim = embed_dims // 2
        c3_embed_dim = embed_dims

        # Replace PatchEmbedInit and PatchEmbeddingStage with MS_DownSampling
        patch_embed1 = MS_DownSampling(in_channels=in_channels,
                                           embed_dims=c1_embed_dim,
                                           kernel_size=3, stride=1, padding=1, first_layer=True)

        # patch_embed1 = PatchEmbedInit(img_size_h=img_size_h,
        #                                    img_size_w=img_size_w,
        #                                    patch_size=patch_size,
        #                                    in_channels=in_channels,
        #                                    embed_dims=embed_dims // 4)

        stage1 = nn.ModuleList([stage1_class(
            dim=embed_dims // 4, num_heads=num_heads[0], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(1)])

        # patch_embed2 = PatchEmbeddingStage(img_size_h=img_size_h,
        #                                    img_size_w=img_size_w,
        #                                    patch_size=patch_size,
        #                                    in_channels=in_channels,
        #                                    embed_dims=embed_dims // 2)
        
        patch_embed2 = MS_DownSampling(in_channels=c1_embed_dim,
                                           embed_dims=c2_embed_dim,
                                           kernel_size=3, stride=2, padding=1, first_layer=False)

        stage2 = nn.ModuleList([stage2_class(
            dim=embed_dims // 2, num_heads=num_heads[1], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(1)])

        # patch_embed3 = PatchEmbeddingStage(img_size_h=img_size_h,
        #                                    img_size_w=img_size_w,
        #                                    patch_size=patch_size,
        #                                    in_channels=in_channels,
        #                                    embed_dims=embed_dims)
        
        patch_embed3 = MS_DownSampling(in_channels=c2_embed_dim,
                                           embed_dims=c3_embed_dim,
                                           kernel_size=3, stride=2, padding=1, first_layer=False)

        stage3 = nn.ModuleList([stage3_class(
            dim=embed_dims, num_heads=num_heads[2], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths - 2)])

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"patch_embed3", patch_embed3)
        setattr(self, f"stage1", stage1)
        setattr(self, f"stage2", stage2)
        setattr(self, f"stage3", stage3)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        stage1 = getattr(self, f"stage1")
        patch_embed1 = getattr(self, f"patch_embed1")
        stage2 = getattr(self, f"stage2")
        patch_embed2 = getattr(self, f"patch_embed2")
        stage3 = getattr(self, f"stage3")
        patch_embed3 = getattr(self, f"patch_embed3")

        x = patch_embed1(x)
        for blk in stage1:
            x, attn_spikes_one, mlp_spikes_one = blk(x, spike_rate=self.spike_rate)

        x = patch_embed2(x)
        for blk in stage2:
            x, attn_spikes_two, mlp_spikes_two = blk(x, spike_rate=self.spike_rate)
        

        x = patch_embed3(x)
        for blk in stage3:
            x, _, _ = blk(x)

        return x.flatten(3).mean(3), (attn_spikes_one, mlp_spikes_one, attn_spikes_two, mlp_spikes_two)

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x, all_spikes = self.forward_features(x)
        x = self.head(x.mean(0))

        return x, all_spikes


@register_model
def QKFormer(pretrained=False, **kwargs):
    valid_args = [
        'img_size_h', 'img_size_w', 'patch_size', 'in_channels', 'num_classes',
        'embed_dims', 'num_heads', 'mlp_ratios', 'qkv_bias', 'qk_scale',
        'drop_rate', 'attn_drop_rate', 'drop_path_rate', 'norm_layer',
        'depths', 'sr_ratios', 'T', 'pretrained_cfg', 'transformer_classes',
        'spike_rate'
    ]

    # Filter kwargs to include only valid arguments
    kwargs = {key: value for key, value in kwargs.items() if key in valid_args}

    # Now pass the filtered arguments to spiking_transformer
    model = spiking_transformer(
        **kwargs  # Pass the filtered arguments
    )

    model.default_cfg = _cfg()

    return model


@register_model
def QKFormer_10_384(T=1, **kwargs):
    model = spiking_transformer(
        T=T,
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=384, num_heads=6, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=10, sr_ratios=1,
        **kwargs
    )

    model.default_cfg = _cfg()

    return model

@register_model
def QKFormer_10_768(T=1, **kwargs):
    model = spiking_transformer(
        T=T,
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=768, num_heads=12, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=10, sr_ratios=1,
        **kwargs
    )
    print(f"We are working with {T} time steps")
    model.default_cfg = _cfg()
    return model



if __name__ == '__main__':
    input = torch.randn(2, 3, 32, 32).cuda()
    model = create_model(
        'QKFormer',
        pretrained=False,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=384, num_heads=8, mlp_ratios=4,
        in_channels=3, num_classes=100, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
        T=4,
    ).cuda()

    from torchinfo import summary

    summary(model, input_size=(2, 3, 32, 32))