
import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
    
"""class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )
        # Residual shortcut — 1x1 conv if channels differ
        self.shortcut = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)"""


class Down3D(nn.Module):
    """MaxPool3d then ConvBlock3D"""
    def __init__(self, in_ch, out_ch, pool_kernel=(2, 2, 2)):
        super().__init__()
        self.pool = nn.MaxPool3d(pool_kernel)
        self.conv = ConvBlock3D(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class AttentionGate3D(nn.Module):
   
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        # Gating signal projection
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, bias=False),
            nn.InstanceNorm3d(F_int, affine=True),
        )
        # Skip connection projection
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, bias=False),
            nn.InstanceNorm3d(F_int, affine=True),
        )
        # Attention coefficient
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, bias=False),
            nn.InstanceNorm3d(1, affine=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:],
                              mode='trilinear', align_corners=False)
        g1  = self.W_g(g)
        x1  = self.W_x(x)
        psi = self.psi(self.relu(g1 + x1))  
        return x * psi                      


class Up3D(nn.Module):
    
    def __init__(self, in_ch, skip_ch, out_ch, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionGate3D(
                F_g=in_ch,
                F_l=skip_ch,
                F_int=skip_ch // 2,
            )
        self.conv = ConvBlock3D(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:],
                          mode='trilinear', align_corners=False)
        if self.use_attention:
            skip = self.attention(x, skip)
        return self.conv(torch.cat([x, skip], dim=1))


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 filters: tuple = (32, 64, 128, 256, 512),
                 dropout: float = 0.3,
                 use_attention: bool = True,
                 deep_supervision: bool = False):
        super().__init__()
        f = filters
        self.deep_supervision = deep_supervision

        # ── Encoder ──────────────────────────────────────────────
        self.enc1 = ConvBlock3D(in_channels, f[0])
        self.enc2 = Down3D(f[0], f[1], pool_kernel=(1, 2, 2))
        self.enc3 = Down3D(f[1], f[2], pool_kernel=(2, 2, 2))
        self.enc4 = Down3D(f[2], f[3], pool_kernel=(2, 2, 2))

        # ── Bottleneck ───────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            Down3D(f[3], f[4], pool_kernel=(2, 2, 2)),
            nn.Dropout3d(dropout),
        )

        # ── Decoder ──────────────────────────────────────────────
        self.dec4 = Up3D(f[4], f[3], f[3], use_attention=use_attention)
        self.dec3 = Up3D(f[3], f[2], f[2], use_attention=use_attention)
        self.dec2 = Up3D(f[2], f[1], f[1], use_attention=use_attention)
        self.dec1 = Up3D(f[1], f[0], f[0], use_attention=use_attention)

        # ── Main head ────────────────────────────────────────────
        self.head = nn.Conv3d(f[0], out_channels, kernel_size=1)

        # if deep_supervision=True
        if deep_supervision:
            self.aux_head2 = nn.Conv3d(f[1], out_channels, kernel_size=1)
            self.aux_head3 = nn.Conv3d(f[2], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b  = self.bottleneck(s4)
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        out_main = self.head(d1)

        if self.deep_supervision:
            out2 = F.interpolate(
                self.aux_head2(d2),
                size=out_main.shape[2:],
                mode='trilinear', align_corners=False
        )
            out3 = F.interpolate(
                self.aux_head3(d3),
                size=out_main.shape[2:],
                mode='trilinear', align_corners=False
        )
            if self.training:
                return out_main, out2, out3
            else:
            # Ensemble aux predictions at inference — helps small nodules
                return (out_main + 0.4 * out2 + 0.2 * out3) / 1.6

        return out_main    

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == "__main__":
    m = UNet3D(use_attention=True)
    x = torch.randn(2, 1, 32, 64, 64)
    y = m(x)
    print(f"UNet3D (attention) | input {tuple(x.shape)} → output {tuple(y.shape)}")
    print(f"UNet3D (attention) | parameters: {count_parameters(m):,}")

  
    m_orig = UNet3D(use_attention=False)
    y_orig = m_orig(x)
    print(f"\nUNet3D (original)  | input {tuple(x.shape)} → output {tuple(y_orig.shape)}")
    print(f"UNet3D (original)  | parameters: {count_parameters(m_orig):,}")

    # Parameter difference = attention gate parameters only
    diff = count_parameters(m) - count_parameters(m_orig)
    print(f"\nAttention gate parameters added: {diff:,}")