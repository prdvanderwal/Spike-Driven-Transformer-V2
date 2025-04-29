import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial

# Based on available information about SpiLiFormer:
# - Three-stage hierarchical architecture
# - Incorporates lateral inhibition through FF-LiDiff and FB-LiDiff mechanisms
# - FF-LiDiff for shallow blocks, FB-LiDiff for deep blocks
# - Uses 4×4 patch in stage 1, and 2×2 patch in stages 2 and 3
# - Neuromorphic hardware friendly with Conv1d instead of RepConv
# - Has fixed number of heads

class MS_MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        return x

class MS_Attention_pushpull(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")


        # Standard query processing
        self.q_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(dim)
        )
        
        self.qon_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.qoff_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.qpp_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.push_w = nn.Parameter(torch.tensor(1.0))
        self.pull_w = nn.Parameter(torch.tensor(1.0))
        self.softplus = nn.Softplus()
        self.pp_bias = nn.Parameter(torch.tensor(0.0))

        self.k_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(dim)
        )
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.v_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(dim)
        )
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")

        self.proj_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        x = self.head_lif(x)
        x_flat = x.flatten(3)  # [T, B, C, H*W]

        # Standard attention mechanism (SDSA-1)
        q = self.q_conv(x_flat.flatten(0, 1))  # [T*B, C, H*W]
        q = q.reshape(T, B, C, N)
        
        k = self.k_conv(x_flat.flatten(0, 1))  # [T*B, C, H*W]
        k = k.reshape(T, B, C, N)
        
        v = self.v_conv(x_flat.flatten(0, 1))  # [T*B, C, H*W]
        v = v.reshape(T, B, C, N)

        q_on = self.qon_lif(q)
        q_off = self.qoff_lif(-q)

        q_on = (
            q_on.transpose(-1, -2)  # -> (T, B, N, C)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()  # -> (T, B, H, N, C/H)
        )

        # and reshape *q_off*
        q_off = (
            q_off.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        
        k = self.k_lif(k)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_lif(v)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        pw = self.softplus(self.push_w)
        pl = self.softplus(self.pull_w)

        diff = pw * q_on - pl * q_off + self.pp_bias

        q_pp = self.qpp_lif(diff)

        x = k.transpose(-2, -1) @ v
        x = (q_pp @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.reshape(T, B, C, H, W)
        
        # Project using Conv1d
        x_flat = x.flatten(3)  # [T, B, C, H*W]
        x_flat = self.proj_conv(x_flat.flatten(0, 1))  # [T*B, C, H*W]
        x = x_flat.reshape(T, B, C, H, W)

        return x



class MS_Attention_linear(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")


        # Standard query processing
        self.q_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(dim)
        )
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.k_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(dim)
        )
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.v_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(dim)
        )
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")

        self.proj_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        x = self.head_lif(x)
        x_flat = x.flatten(3)  # [T, B, C, H*W]

        # Standard attention mechanism (SDSA-1)
        q = self.q_conv(x_flat.flatten(0, 1))  # [T*B, C, H*W]
        q = q.reshape(T, B, C, N)
        
        k = self.k_conv(x_flat.flatten(0, 1))  # [T*B, C, H*W]
        k = k.reshape(T, B, C, N)
        
        v = self.v_conv(x_flat.flatten(0, 1))  # [T*B, C, H*W]
        v = v.reshape(T, B, C, N)

        q = self.q_lif(q)
        q = (
            q.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        
        k = self.k_lif(k)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_lif(v)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.reshape(T, B, C, H, W)
        
        # Project using Conv1d
        x_flat = x.flatten(3)  # [T, B, C, H*W]
        x_flat = self.proj_conv(x_flat.flatten(0, 1))  # [T*B, C, H*W]
        x = x_flat.reshape(T, B, C, H, W)

        return x


class MS_Attention_inhibition(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        # For FF-LiDiff (Feedforward Lateral Differential Inhibition)
        # Split query into excitatory and inhibitory pathways
        self.q_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(dim)
        )

        self.qe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.qi_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.merge_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
       
        self.k_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(dim)
        )
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.attn_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.proj_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        x = self.head_lif(x)
        x_flat = x.flatten(3)  # [T, B, C, H*W]


        # FF-LiDiff implementation
        qe, qi = self.q_conv(x_flat.flatten(0, 1)).chunk(2, dim=1)  # [T*B, C/2, H*W]
        qe = qe.reshape(T, B, C//2, N)
        qi = qi.reshape(T, B, C//2, N)
        
        k = self.k_conv(x_flat.flatten(0, 1))  # [T*B, C, H*W]
        k = k.reshape(T, B, C, N)
        

        qe = self.qe_lif(qe)
        qe = (
            qe.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // (2 * self.num_heads))
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        
        qi = self.qi_lif(qi)
        qi = (
            qi.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // (2 * self.num_heads))
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        
        k = self.k_lif(k)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        # Sum over channel dimension
        qi = qi.sum(dim=-1).unsqueeze(-1)
        qe = qe.sum(dim=-1).unsqueeze(-1)

        # Calculate elementwise difference
        filtered_q = self.merge_lif(qe-qi)

        # Multiply filtered q with k and reshape
        x = filtered_q * k

        # Pass combined through last LIF
        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.reshape(T, B, C, H, W)
        
        # Project using Conv1d
        x_flat = x.flatten(3)  # [T, B, C, H*W]
        x_flat = self.proj_conv(x_flat.flatten(0, 1))  # [T*B, C, H*W]
        x = x_flat.reshape(T, B, C, H, W)

        return x



class MS_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        lateral_inhibition=False,
        use_feedback=False,
        push_pull=False,
    ):
        super().__init__()
        self.use_feedback = use_feedback
        
        if lateral_inhibition:

            if push_pull:
                self.attn = MS_Attention_pushpull(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
            )
            else:
                # FF-LiDiff for shallow blocks or FB-LiDiff for deep blocks
                self.attn = MS_Attention_inhibition(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    sr_ratio=sr_ratio,
                )
        else:
            self.attn = MS_Attention_linear(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
            )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        # Note: For FB-LiDiff (Feedback Lateral Differential Inhibition)
        # This would be implemented here with feedback connections
        # Comment: Add FB-LiDiff implementation here for deep blocks
        if self.use_feedback:
            # FB-LiDiff would be implemented here
            # This would require maintaining state between timesteps
            pass
            
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        
        return x


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


class Spiking_vit_SpiLiFormer(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dim=[64, 128, 256],  # Dimensions for each stage
        num_heads=8,  # Fixed number of heads as in Spike-driven Transformer V2
        mlp_ratios=[4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[1, 1, 2],
        sr_ratios=[8, 4, 2],
        T=4,
        push_pull=False,
        lateral_inhibition=False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        
        # Stage 1: C × H/4 × W/4
        # Stage 2: 2C × H/8 × W/8
        # Stage 3: 4C × H/16 × W/16
        
        # Initial downsampling with 4×4 patch size (kernel_size=4, stride=4)
        self.downsample1 = MS_DownSampling(
            in_channels=in_chans,
            embed_dims=embed_dim[0],
            kernel_size=4,  # 4×4 patch as specified in the paper
            stride=4,       # Stride of 4 to get H/4 × W/4
            padding=0,
            first_layer=True,
        )
        
        # Stage 1
        self.stage1_blocks = nn.ModuleList([
            MS_Block(
                dim=embed_dim[0],
                num_heads=num_heads,
                mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[0],
                lateral_inhibition=lateral_inhibition,  # FF-LiDiff for shallow blocks
                use_feedback=False,
                push_pull=push_pull
            )
            for i in range(depths[0])
        ])
        
        # Downsampling from Stage 1 to Stage 2 with 2×2 patch size
        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=2,  # 2×2 patch as specified in the paper
            stride=2,       # Stride of 2 to get H/8 × W/8
            padding=0,
            first_layer=False,
        )
        
        # Blocks for Stage 2 with FF-LiDiff
        self.stage2_blocks = nn.ModuleList([
            MS_Block(
                dim=embed_dim[1],
                num_heads=num_heads,
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[1],
                lateral_inhibition=lateral_inhibition,  # FF-LiDiff for shallow blocks
                use_feedback=False,
                push_pull=push_pull,
            )
            for i in range(depths[1])
        ])
        
        # Downsampling from Stage 2 to Stage 3 with 2×2 patch size
        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=2,  # 2×2 patch as specified in the paper
            stride=2,       # Stride of 2 to get H/16 × W/16
            padding=0,
            first_layer=False,
        )
        
        # Blocks for Stage 3 with FB-LiDiff
        self.stage3_blocks = nn.ModuleList([
            MS_Block(
                dim=embed_dim[2],
                num_heads=num_heads,
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[2],
                lateral_inhibition=lateral_inhibition,  # FB-LiDiff for deep blocks
                use_feedback=False, # Enable feedback for FB-LiDiff
                push_pull=push_pull       
            )
            for i in range(depths[2])
        ])

        # Classification Head
        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(embed_dim[2], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # Convert input to shape [T, B, C, H, W] for multi-timestep processing
        T = self.T
        B, C, H, W = x.shape
        x = x.repeat(T, 1, 1, 1, 1)  # [T, B, C, H, W]

        # Stage 1: C × H/4 × W/4
        x = self.downsample1(x)  # Using MS_DownSampling with 4×4 kernel
        for blk in self.stage1_blocks:
            x = blk(x)
        
        # Stage 2: 2C × H/8 × W/8
        x = self.downsample2(x)  # Using MS_DownSampling with 2×2 kernel
        for blk in self.stage2_blocks:
            x = blk(x)
        
        # Stage 3: 4C × H/16 × W/16
        x = self.downsample3(x)  # Using MS_DownSampling with 2×2 kernel
        for blk in self.stage3_blocks:
            x = blk(x)
        
        return x

    def forward(self, x):
        x = self.forward_features(x)
        
        # Global Average Pooling
        x = self.head_lif(x)  # Apply spike activation
        x = self.avgpool(x.flatten(0, 1)).reshape(self.T, -1, x.size(2))  # [T, B, C]
        x = x.mean(0)  # Average over timesteps [B, C]
        
        x = self.head(x)
        return x


@register_model
def spiliformer_tiny(pretrained=False, **kwargs):
    model = Spiking_vit_SpiLiFormer(
        embed_dim=[384,384,384],
        num_heads=8,  # Fixed number of heads
        mlp_ratios=[4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[1,1,2],
        sr_ratios=[8, 4, 2],
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def spiliformer_small(pretrained=False, **kwargs):
    model = Spiking_vit_SpiLiFormer(
        embed_dim=[96, 192, 384],
        num_heads=8,  # Fixed number of heads
        mlp_ratios=[4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 12],
        sr_ratios=[8, 4, 2],
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def spiliformer_base(pretrained=False, **kwargs):
    model = Spiking_vit_SpiLiFormer(
        embed_dim=[128, 256, 512],
        num_heads=8,  # Fixed number of heads
        mlp_ratios=[4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 18],
        sr_ratios=[8, 4, 2],
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
