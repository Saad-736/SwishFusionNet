"""
models.py
---------
Models for the final-exam comparative study.

Three EXTERNAL baselines (re-implementations of the three survey papers):

  1. alexnet     — Paper 1, Yanai & Kawano, ICMEW 2015
  2. mresnet50   — Paper 2, Abdul Kareem et al., CBM 2024 (ResNet-50 + Swish)
  3. vit         — Paper 3, Ghosh & Sazonov, EMBC 2025 (ViT-B/16 + noise)

Five PROPOSED variants for the ablation study (all share one class
`SwishFusionNet`, toggled by config flags):

  V1 v1_resnet50      — Vanilla ResNet-50 (ReLU, no transformer, no noise)
  V2 v2_swish         — + Swish activations
  V3 v3_transformer   — + Swish + Transformer head over spatial tokens
  V4 v4_fixed_noise   — + fixed scalar noise injection (NoisyViT-style)
  V5 v5_proposed      — + ADAPTIVE noise: per-channel learnable σ
                        gated by prediction entropy (THE NOVELTY)

Novel contribution of the proposed model:
  • Hybrid CNN–Transformer specifically for fine-grained food images.
  • Adaptive Noise Injection (ANI): the noise standard deviation is
    learned per channel, and a global gate scales it by the soft-prediction
    entropy of the same forward pass. Confident batches get less noise;
    uncertain (early-training) batches get more — a learned curriculum
    regularizer, unlike NoisyViT's fixed scalar σ.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _swap_relu_with_silu(module: nn.Module) -> None:
    """Recursively replace every nn.ReLU with nn.SiLU (Swish)."""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.SiLU(inplace=True))
        else:
            _swap_relu_with_silu(child)


# ===========================================================================
# EXTERNAL BASELINES
# ===========================================================================
def build_alexnet(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Paper 1: AlexNet pre-trained on ImageNet, fine-tuned on Food-101."""
    weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.alexnet(weights=weights)
    in_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_features, num_classes)
    return net


class MResNet50(nn.Module):
    """Paper 2: ResNet-50 with Swish activations + light pruning proxy."""
    def __init__(self, num_classes: int, pretrained: bool = True,
                 stage2_drop: float = 0.08, stage3_drop: float = 0.10):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        _swap_relu_with_silu(self.backbone)
        self.drop2 = nn.Dropout2d(p=stage2_drop)
        self.drop3 = nn.Dropout2d(p=stage3_drop)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        b = self.backbone
        x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x); x = self.drop2(x)
        x = b.layer3(x); x = self.drop3(x)
        x = b.layer4(x)
        x = b.avgpool(x); x = torch.flatten(x, 1)
        x = b.fc(x)
        return x


def build_mresnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    return MResNet50(num_classes=num_classes, pretrained=pretrained)


class NoisyViT(nn.Module):
    """Paper 3: ViT-B/16 with linear-transform noise injection in last block."""
    def __init__(self, num_classes: int, pretrained: bool = True,
                 noise_std: float = 0.05):
        super().__init__()
        import timm
        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes,
        )
        self.noise_std = noise_std
        self.noise_gate = nn.Parameter(torch.tensor(1.0))
        self.vit.blocks[-1].register_forward_pre_hook(self._noise_hook)

    def _noise_hook(self, module, inputs):
        if not self.training:
            return None
        x = inputs[0]
        noise = torch.randn_like(x) * self.noise_std
        return ((x + self.noise_gate * noise),) + inputs[1:]

    def forward(self, x):
        return self.vit(x)


def build_vit(num_classes: int, pretrained: bool = True) -> nn.Module:
    return NoisyViT(num_classes=num_classes, pretrained=pretrained)


# ===========================================================================
# PROPOSED MODEL: SwishFusionNet (with ablation flags)
# ===========================================================================
class AdaptiveNoise(nn.Module):
    """Adaptive Noise Injection (ANI) module.

    Args:
        feat_dim: dimensionality of the feature vector to perturb.
        init_log_std: initial value of per-channel log-σ (default -3.0 → σ≈0.05).

    Operation (training only):
        1. Predict soft logits from the input features (detached gradient).
        2. Compute Shannon entropy of softmax(logits); normalize by log(C).
        3. Pass normalized entropy through a small MLP gate → scalar in (0,1).
        4. Add noise: x ← x + ε * σ_channel * gate,
           where σ_channel = exp(log_std), per-channel learnable.
    """
    def __init__(self, feat_dim: int, num_classes: int,
                 init_log_std: float = -3.0):
        super().__init__()
        self.log_std = nn.Parameter(torch.full((feat_dim,), init_log_std))
        self.entropy_gate = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )
        # Auxiliary classifier head used only to estimate prediction entropy.
        # Detached from the main grad path.
        self.aux_classifier = nn.Linear(feat_dim, num_classes)
        self.register_buffer("max_entropy", torch.tensor(math.log(num_classes)))

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return feat, torch.zeros(1, device=feat.device)
        with torch.no_grad():
            logits = self.aux_classifier(feat.detach())
            probs = F.softmax(logits, dim=-1)
            ent = -(probs * (probs + 1e-9).log()).sum(dim=-1, keepdim=True)
            ent_norm = (ent / self.max_entropy).clamp(0.0, 1.0)
        gate = self.entropy_gate(ent_norm)           # B x 1, in (0, 1)
        sigma = self.log_std.exp().unsqueeze(0)      # 1 x C
        noise = torch.randn_like(feat) * sigma * gate
        return feat + noise, gate.mean().detach()


class SwishFusionNet(nn.Module):
    """Hybrid CNN-Transformer for fine-grained food recognition.

    Configurable via flags so the same class implements all ablation
    variants V1-V5:

        use_swish               True  → Swish activations in backbone
        use_transformer_head    True  → 2-layer self-attention over 7×7 tokens
        noise_mode              'none' | 'fixed' | 'adaptive'
    """
    def __init__(self,
                 num_classes: int,
                 use_swish: bool = True,
                 use_transformer_head: bool = True,
                 noise_mode: str = "adaptive",
                 transformer_depth: int = 2,
                 transformer_heads: int = 8,
                 token_dim: int = 512,
                 fixed_noise_std: float = 0.05,
                 pretrained: bool = True):
        super().__init__()
        assert noise_mode in {"none", "fixed", "adaptive"}
        self.use_transformer_head = use_transformer_head
        self.noise_mode = noise_mode
        self.fixed_noise_std = fixed_noise_std

        # ---- Backbone: ResNet-50 up to layer4 (output 2048 x 7 x 7)
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        if use_swish:
            _swap_relu_with_silu(backbone)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )

        # ---- Optional Transformer head over spatial tokens
        if use_transformer_head:
            self.proj = nn.Conv2d(2048, token_dim, kernel_size=1)
            self.pos_embed = nn.Parameter(torch.randn(1, 49, token_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=token_dim, nhead=transformer_heads,
                dim_feedforward=token_dim * 2, dropout=0.1,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=transformer_depth)
            feat_dim = token_dim
        else:
            feat_dim = 2048

        # ---- Noise module
        if noise_mode == "adaptive":
            self.noise_mod = AdaptiveNoise(feat_dim, num_classes)
        else:
            self.noise_mod = None

        # ---- Classification head
        self.classifier = nn.Linear(feat_dim, num_classes)

        # Diagnostic: store last gate value for logging
        self.last_gate = torch.tensor(0.0)

    def forward(self, x):
        feat = self.stem(x)  # B x 2048 x 7 x 7
        if self.use_transformer_head:
            feat = self.proj(feat)                       # B x C x 7 x 7
            tokens = feat.flatten(2).transpose(1, 2)     # B x 49 x C
            tokens = tokens + self.pos_embed
            tokens = self.transformer(tokens)            # B x 49 x C
            feat_vec = tokens.mean(dim=1)                # B x C  (token-avg)
        else:
            feat_vec = feat.mean(dim=[2, 3])             # B x 2048 (global-avg)

        # Noise injection (training only)
        if self.training:
            if self.noise_mode == "fixed":
                feat_vec = feat_vec + torch.randn_like(feat_vec) * self.fixed_noise_std
            elif self.noise_mode == "adaptive":
                feat_vec, gate = self.noise_mod(feat_vec)
                self.last_gate = gate

        return self.classifier(feat_vec)


# ---------------------------------------------------------------------------
# Variant builders
# ---------------------------------------------------------------------------
def build_v1_resnet50(num_classes, pretrained=True):
    return SwishFusionNet(num_classes,
                          use_swish=False, use_transformer_head=False,
                          noise_mode="none", pretrained=pretrained)

def build_v2_swish(num_classes, pretrained=True):
    return SwishFusionNet(num_classes,
                          use_swish=True, use_transformer_head=False,
                          noise_mode="none", pretrained=pretrained)

def build_v3_transformer(num_classes, pretrained=True):
    return SwishFusionNet(num_classes,
                          use_swish=True, use_transformer_head=True,
                          noise_mode="none", pretrained=pretrained)

def build_v4_fixed_noise(num_classes, pretrained=True):
    return SwishFusionNet(num_classes,
                          use_swish=True, use_transformer_head=True,
                          noise_mode="fixed", pretrained=pretrained)

def build_v5_proposed(num_classes, pretrained=True):
    return SwishFusionNet(num_classes,
                          use_swish=True, use_transformer_head=True,
                          noise_mode="adaptive", pretrained=pretrained)


# ===========================================================================
# Registry
# ===========================================================================
MODEL_BUILDERS = {
    # External baselines (the three survey papers)
    "alexnet":         build_alexnet,
    "mresnet50":       build_mresnet50,
    "vit":             build_vit,
    # Ablation variants of the proposed model
    "v1_resnet50":     build_v1_resnet50,
    "v2_swish":        build_v2_swish,
    "v3_transformer":  build_v3_transformer,
    "v4_fixed_noise":  build_v4_fixed_noise,
    "v5_proposed":     build_v5_proposed,
}


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](num_classes=num_classes, pretrained=pretrained)


if __name__ == "__main__":
    for name in MODEL_BUILDERS:
        m = build_model(name, num_classes=20, pretrained=False)
        n = count_parameters(m) / 1e6
        print(f"{name:18s}  params = {n:6.2f} M")
