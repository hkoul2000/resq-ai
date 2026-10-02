"""ResQNet: Multimodal flood prediction network with uncertainty quantification.

Architecture:
- SAR/Optical encoder: U-Net with ResNet-34 backbone (or SegFormer-B0 variant)
- Static-geo branch: small CNN over terrain features
- Rainfall branch: 1D temporal conv + Transformer encoder
- Fusion: FiLM conditioning + gated cross-attention at bottleneck
- Modality dropout for missing-modality robustness
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ==============================================================================
# Building blocks
# ==============================================================================


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer.

    Applies affine transformation: gamma * x + beta
    where gamma and beta are predicted from a conditioning vector.
    """

    def __init__(self, feature_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, feature_dim)
        self.beta_proj = nn.Linear(cond_dim, feature_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            x: Feature map (B, C, H, W) or (B, C)
            cond: Conditioning vector (B, cond_dim)
        """
        gamma = self.gamma_proj(cond)  # (B, C)
        beta = self.beta_proj(cond)  # (B, C)

        if x.dim() == 4:
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class GatedCrossAttention(nn.Module):
    """Gated cross-attention for multimodal fusion.

    Allows selective attention between modalities with a learned gate.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-attend from query to key_value with gating.

        Args:
            query: (B, N_q, D)
            key_value: (B, N_kv, D)
            mask: optional boolean mask for key_value availability
        """
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)

        attn_out, _ = self.attention(q, kv, kv, key_padding_mask=mask)
        gate = self.gate(attn_out)
        return query + gate * attn_out


class ModalityGate(nn.Module):
    """Learned gate for each modality, controlled by availability mask."""

    def __init__(self, dim: int, num_modalities: int) -> None:
        super().__init__()
        self.gates = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid()) for _ in range(num_modalities)]
        )

    def forward(
        self, features: list[torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        """Combine features with learned gates, respecting availability mask.

        Args:
            features: list of (B, D) tensors, one per modality
            mask: (B, num_modalities) availability mask
        """
        gated = []
        for i, (feat, gate) in enumerate(zip(features, self.gates)):
            g = gate(feat) * mask[:, i : i + 1]  # Zero out unavailable modalities
            gated.append(g * feat)

        return sum(gated)


# ==============================================================================
# Branch encoders
# ==============================================================================


class GeoEncoder(nn.Module):
    """Small CNN encoder for static geographic features.

    Input: (B, C_geo, H, W) where C_geo includes elevation, slope, HAND,
    TWI, land cover one-hot, distance to drainage.
    Output: (B, out_dim) global feature vector
    """

    def __init__(self, in_channels: int = 6, out_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class GeoEncoderSpatial(nn.Module):
    """Spatial geo encoder that preserves spatial dimensions for skip connections.

    Output: (B, out_dim, H/4, W/4) spatial feature map
    """

    def __init__(self, in_channels: int = 6, out_dim: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class RainfallEncoder(nn.Module):
    """1D temporal convolution + Transformer encoder for rainfall sequences.

    Input: (B, seq_len, 1) daily rainfall values
    Output: (B, out_dim) rainfall feature vector
    """

    def __init__(
        self,
        seq_len: int = 30,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        out_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # 1D temporal convolution to embed the sequence
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process rainfall sequence.

        Args:
            x: (B, seq_len, 1) or (B, seq_len)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, seq_len, 1)

        # Conv expects (B, C, L)
        x = x.permute(0, 2, 1)  # (B, 1, seq_len)
        x = self.temporal_conv(x)  # (B, d_model, seq_len)
        x = x.permute(0, 2, 1)  # (B, seq_len, d_model)

        # Add positional encoding
        x = x + self.pos_encoding[:, : x.size(1), :]

        # Transformer
        x = self.transformer(x)  # (B, seq_len, d_model)

        # Global average pooling over time
        x = x.mean(dim=1)  # (B, d_model)

        return self.fc(x)


# ==============================================================================
# U-Net decoder
# ==============================================================================


class DecoderBlock(nn.Module):
    """U-Net decoder block with skip connection."""

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.up(x)

        if skip is not None:
            # Handle size mismatch
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)


# ==============================================================================
# ResQNet main model
# ==============================================================================


class ResQNet(nn.Module):
    """ResQNet: Uncertainty-aware multimodal flood prediction network.

    Architecture overview:
    1. SAR/Optical shared U-Net encoder (ResNet-34 backbone)
    2. Static-geo branch (small CNN)
    3. Rainfall branch (Conv1D + Transformer)
    4. FiLM conditioning + Gated cross-attention fusion at bottleneck
    5. U-Net decoder with skip connections
    6. Output: flood probability per pixel

    Supports modality dropout during training for robustness.
    """

    def __init__(
        self,
        sar_channels: int = 2,
        optical_channels: int = 13,
        geo_channels: int = 6,
        rainfall_seq_len: int = 30,
        rainfall_d_model: int = 64,
        rainfall_nhead: int = 4,
        rainfall_layers: int = 2,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        fusion: str = "film_crossattention",
        dropout: float = 0.2,
        modality_dropout: float = 0.2,
        num_classes: int = 1,
        evidential: bool = False,
    ) -> None:
        super().__init__()

        self.sar_channels = sar_channels
        self.optical_channels = optical_channels
        self.fusion_type = fusion
        self.modality_dropout_p = modality_dropout
        self.evidential = evidential
        self.num_classes = num_classes

        # Bottleneck dimension
        self.bottleneck_dim = 512  # ResNet-34 last layer

        # ---- SAR encoder ----
        self._build_encoder(encoder_name, sar_channels, optical_channels, pretrained)

        # ---- Geo encoder ----
        self.geo_encoder = GeoEncoder(in_channels=geo_channels, out_dim=128)
        self.geo_encoder_spatial = GeoEncoderSpatial(in_channels=geo_channels, out_dim=64)

        # ---- Rainfall encoder ----
        self.rainfall_encoder = RainfallEncoder(
            seq_len=rainfall_seq_len,
            d_model=rainfall_d_model,
            nhead=rainfall_nhead,
            num_layers=rainfall_layers,
            out_dim=128,
            dropout=dropout,
        )

        # ---- Fusion modules ----
        cond_dim = 128 + 128  # geo + rainfall
        self.film = FiLMLayer(self.bottleneck_dim, cond_dim)

        if fusion == "film_crossattention":
            self.cross_attention = GatedCrossAttention(
                dim=self.bottleneck_dim, num_heads=4, dropout=dropout
            )
            # Project conditioning to bottleneck dim for cross-attention
            self.cond_proj = nn.Linear(cond_dim, self.bottleneck_dim)

        # ---- Modality gate ----
        self.modality_gate = ModalityGate(self.bottleneck_dim, num_modalities=2)
        # SAR and optical at the image encoder level

        # ---- Decoder ----
        # ResNet-34 encoder channel sizes: [64, 64, 128, 256, 512]
        encoder_channels = [64, 64, 128, 256, 512]
        decoder_channels = [256, 128, 64, 32]

        self.decoder_blocks = nn.ModuleList()
        in_ch = self.bottleneck_dim
        for i, out_ch in enumerate(decoder_channels):
            skip_ch = encoder_channels[-(i + 2)] if i < len(encoder_channels) - 1 else 0
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        # Final upsampling to match input resolution
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[-1], 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        # ---- Output head ----
        if evidential:
            # Evidential: output 4 params for Beta distribution (alpha, beta)
            # or Dirichlet for multi-class
            self.head = nn.Conv2d(16, 4, 1)  # alpha, beta, lambda, nu for NIG
        else:
            self.head = nn.Conv2d(16, num_classes, 1)

        self.dropout = nn.Dropout2d(dropout)

    def _build_encoder(
        self,
        encoder_name: str,
        sar_channels: int,
        optical_channels: int,
        pretrained: bool,
    ) -> None:
        """Build the image encoder using timm."""
        try:
            import timm

            # Create separate encoders for SAR and optical to handle different input channels
            self.sar_encoder = timm.create_model(
                encoder_name,
                pretrained=pretrained,
                features_only=True,
                in_chans=sar_channels,
                out_indices=(0, 1, 2, 3, 4),
            )
            self.optical_encoder = timm.create_model(
                encoder_name,
                pretrained=pretrained,
                features_only=True,
                in_chans=optical_channels,
                out_indices=(0, 1, 2, 3, 4),
            )
            logger.info(f"Built {encoder_name} encoders (SAR: {sar_channels}ch, Optical: {optical_channels}ch)")
        except ImportError:
            logger.warning("timm not available, building simple encoder")
            self._build_simple_encoder(sar_channels, optical_channels)

    def _build_simple_encoder(
        self, sar_channels: int, optical_channels: int
    ) -> None:
        """Fallback simple encoder without timm."""
        def make_encoder(in_ch: int) -> nn.Module:
            return nn.ModuleList([
                nn.Sequential(nn.Conv2d(in_ch, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True)),
                nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)),
                nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)),
                nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
                nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)),
            ])
        self.sar_encoder = make_encoder(sar_channels)
        self.optical_encoder = make_encoder(optical_channels)

    def _encode_image(
        self,
        encoder: nn.Module,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run image through encoder, return bottleneck + skip features."""
        if hasattr(encoder, "forward_features"):
            # timm model with features_only=True
            features = encoder(x)
            return features[-1], features[:-1]
        else:
            # Simple fallback encoder
            skips = []
            for layer in encoder:
                x = layer(x)
                skips.append(x)
            return skips[-1], skips[:-1]

    def forward(
        self,
        sar: torch.Tensor | None = None,
        optical: torch.Tensor | None = None,
        geo: torch.Tensor | None = None,
        rainfall: torch.Tensor | None = None,
        modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            sar: SAR input (B, 2, H, W)
            optical: Optical input (B, 13, H, W)
            geo: Geographic features (B, C_geo, H, W)
            rainfall: Rainfall sequence (B, seq_len, 1)
            modality_mask: (B, 5) binary mask [sar, optical, dem, landcover, rainfall]

        Returns:
            dict with keys:
                - logits: (B, 1, H, W) raw logits
                - probs: (B, 1, H, W) sigmoid probabilities
                - (if evidential): alpha, beta, evidence
        """
        B = (sar if sar is not None else optical).shape[0]
        device = (sar if sar is not None else optical).device

        if modality_mask is None:
            modality_mask = torch.ones(B, 5, device=device)

        # ---- Image encoding ----
        sar_features = None
        sar_skips = None
        optical_features = None
        optical_skips = None

        if sar is not None and modality_mask[:, 0].sum() > 0:
            sar_features, sar_skips = self._encode_image(self.sar_encoder, sar)

        if optical is not None and modality_mask[:, 1].sum() > 0:
            optical_features, optical_skips = self._encode_image(self.optical_encoder, optical)

        # Combine image features via modality gate
        img_features_list = []
        img_mask = torch.zeros(B, 2, device=device)

        if sar_features is not None:
            img_features_list.append(sar_features)
            img_mask[:, 0] = modality_mask[:, 0]
        else:
            # Zero placeholder
            if optical_features is not None:
                img_features_list.append(torch.zeros_like(optical_features))
            else:
                # Both missing - create zero tensor
                img_features_list.append(torch.zeros(B, self.bottleneck_dim, 1, 1, device=device))
            img_mask[:, 0] = 0

        if optical_features is not None:
            img_features_list.append(optical_features)
            img_mask[:, 1] = modality_mask[:, 1]
        else:
            if sar_features is not None:
                img_features_list.append(torch.zeros_like(sar_features))
            else:
                img_features_list.append(torch.zeros(B, self.bottleneck_dim, 1, 1, device=device))
            img_mask[:, 1] = 0

        # Pool image features to vectors for gating
        h, w = img_features_list[0].shape[-2:]
        pooled_features = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in img_features_list]
        gated_feature_vec = self.modality_gate(pooled_features, img_mask)

        # Combine spatial features (use available encoder's features, or average)
        if sar_features is not None and optical_features is not None:
            # Ensure same spatial size
            if sar_features.shape != optical_features.shape:
                optical_features = F.interpolate(
                    optical_features, size=sar_features.shape[-2:],
                    mode="bilinear", align_corners=False
                )
            bottleneck = (
                sar_features * modality_mask[:, 0:1, None, None]
                + optical_features * modality_mask[:, 1:2, None, None]
            )
        elif sar_features is not None:
            bottleneck = sar_features
        elif optical_features is not None:
            bottleneck = optical_features
        else:
            bottleneck = torch.zeros(B, self.bottleneck_dim, 8, 8, device=device)

        # Select skip connections from available modality
        if sar_skips is not None:
            skips = sar_skips
        elif optical_skips is not None:
            skips = optical_skips
        else:
            skips = [torch.zeros(B, 64, 1, 1, device=device)] * 4

        # ---- Geo encoding ----
        if geo is not None and modality_mask[:, 2].sum() > 0:
            geo_vec = self.geo_encoder(geo)
        else:
            geo_vec = torch.zeros(B, 128, device=device)

        # ---- Rainfall encoding ----
        if rainfall is not None and modality_mask[:, 4].sum() > 0:
            rain_vec = self.rainfall_encoder(rainfall)
        else:
            rain_vec = torch.zeros(B, 128, device=device)

        # ---- Fusion ----
        cond = torch.cat([geo_vec, rain_vec], dim=1)  # (B, 256)

        # FiLM conditioning on bottleneck
        bottleneck = self.film(bottleneck, cond)

        # Cross-attention fusion
        if self.fusion_type == "film_crossattention" and hasattr(self, "cross_attention"):
            # Reshape bottleneck to sequence: (B, C, H, W) -> (B, H*W, C)
            bh, bw = bottleneck.shape[-2:]
            bottleneck_seq = bottleneck.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

            # Project conditioning to bottleneck dim and use as key/value
            cond_proj = self.cond_proj(cond).unsqueeze(1)  # (B, 1, C)
            cond_kv = cond_proj.expand(-1, 4, -1)  # (B, 4, C) - replicate for attention

            bottleneck_seq = self.cross_attention(bottleneck_seq, cond_kv)
            bottleneck = bottleneck_seq.permute(0, 2, 1).reshape(B, -1, bh, bw)

        # ---- Decoder ----
        x = bottleneck
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skips[-(i + 1)] if i < len(skips) else None
            if skip is not None:
                # Ensure spatial sizes match
                if x.shape[-2:] != (skip.shape[-2] // 1, skip.shape[-1] // 1):
                    pass  # DecoderBlock handles upsampling
            x = self.dropout(x)
            x = decoder_block(x, skip)

        # Final upsample
        x = self.final_up(x)

        # ---- Output head ----
        logits = self.head(x)

        output = {"logits": logits}

        if self.evidential:
            # Evidential output: split into evidence parameters
            # Using Normal-Inverse-Gamma parameterization
            gamma = logits[:, 0:1]  # mean
            nu = F.softplus(logits[:, 1:2]) + 1e-6  # virtual observation count
            alpha = F.softplus(logits[:, 2:3]) + 1.0  # shape
            beta = F.softplus(logits[:, 3:4]) + 1e-6  # scale

            output["gamma"] = gamma
            output["nu"] = nu
            output["alpha"] = alpha
            output["beta"] = beta

            # Compute uncertainty decomposition
            aleatoric = beta / (alpha - 1.0 + 1e-8)
            epistemic = aleatoric / (nu + 1e-8)
            output["aleatoric_uncertainty"] = aleatoric
            output["epistemic_uncertainty"] = epistemic
            output["probs"] = torch.sigmoid(gamma)
        else:
            output["probs"] = torch.sigmoid(logits)

        return output


class ResQNetEnsemble(nn.Module):
    """Ensemble of ResQNet models for deep ensemble uncertainty.

    Trains M independent models and combines predictions.
    """

    def __init__(self, ensemble_size: int = 5, **model_kwargs: Any) -> None:
        super().__init__()
        self.ensemble_size = ensemble_size
        self.models = nn.ModuleList(
            [ResQNet(**model_kwargs) for _ in range(ensemble_size)]
        )

    def forward(
        self, member_idx: int | None = None, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            member_idx: If specified, run only that ensemble member.
                       If None, run all members and return aggregated output.
        """
        if member_idx is not None:
            return self.models[member_idx](**kwargs)

        outputs = [model(**kwargs) for model in self.models]

        # Aggregate
        all_probs = torch.stack([o["probs"] for o in outputs], dim=0)  # (M, B, 1, H, W)
        mean_probs = all_probs.mean(dim=0)
        std_probs = all_probs.std(dim=0)

        # Predictive entropy as total uncertainty
        eps = 1e-8
        predictive_entropy = -(
            mean_probs * torch.log(mean_probs + eps)
            + (1 - mean_probs) * torch.log(1 - mean_probs + eps)
        )

        # Aleatoric: mean of individual entropies
        individual_entropies = -(
            all_probs * torch.log(all_probs + eps)
            + (1 - all_probs) * torch.log(1 - all_probs + eps)
        )
        aleatoric = individual_entropies.mean(dim=0)

        # Epistemic: mutual information = predictive entropy - aleatoric
        epistemic = predictive_entropy - aleatoric

        return {
            "probs": mean_probs,
            "logits": outputs[0]["logits"],  # from first member
            "std": std_probs,
            "predictive_entropy": predictive_entropy,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "all_probs": all_probs,
        }


def enable_mc_dropout(model: nn.Module) -> None:
    """Enable dropout layers during inference for MC Dropout."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def mc_dropout_predict(
    model: ResQNet,
    num_samples: int = 20,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    """MC Dropout prediction with uncertainty estimation.

    Args:
        model: ResQNet model
        num_samples: Number of forward passes
        **kwargs: Model inputs

    Returns:
        Dict with mean prediction, uncertainty estimates
    """
    model.eval()
    enable_mc_dropout(model)

    all_probs = []
    with torch.no_grad():
        for _ in range(num_samples):
            output = model(**kwargs)
            all_probs.append(output["probs"])

    all_probs_tensor = torch.stack(all_probs, dim=0)  # (T, B, 1, H, W)
    mean_probs = all_probs_tensor.mean(dim=0)
    std_probs = all_probs_tensor.std(dim=0)

    eps = 1e-8
    predictive_entropy = -(
        mean_probs * torch.log(mean_probs + eps)
        + (1 - mean_probs) * torch.log(1 - mean_probs + eps)
    )

    individual_entropies = -(
        all_probs_tensor * torch.log(all_probs_tensor + eps)
        + (1 - all_probs_tensor) * torch.log(1 - all_probs_tensor + eps)
    )
    aleatoric = individual_entropies.mean(dim=0)
    epistemic = predictive_entropy - aleatoric

    model.train()  # Restore training mode

    return {
        "probs": mean_probs,
        "std": std_probs,
        "predictive_entropy": predictive_entropy,
        "aleatoric_uncertainty": aleatoric,
        "epistemic_uncertainty": epistemic,
        "all_probs": all_probs_tensor,
    }


def tta_predict(
    model: ResQNet,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Test-Time Augmentation prediction.

    Applies flips and 90-degree rotations, averages predictions.
    """
    model.eval()

    def get_image_kwargs(flip_h: bool, flip_v: bool, rot90: int) -> dict:
        """Apply transforms to image inputs."""
        transformed = {}
        for key in ["sar", "optical", "geo"]:
            if key in kwargs and kwargs[key] is not None:
                x = kwargs[key]
                if flip_h:
                    x = torch.flip(x, dims=[-1])
                if flip_v:
                    x = torch.flip(x, dims=[-2])
                if rot90 > 0:
                    x = torch.rot90(x, k=rot90, dims=[-2, -1])
                transformed[key] = x
            else:
                transformed[key] = kwargs.get(key)
        return transformed

    def inverse_transform(
        probs: torch.Tensor, flip_h: bool, flip_v: bool, rot90: int
    ) -> torch.Tensor:
        """Reverse the spatial transforms on the output."""
        if rot90 > 0:
            probs = torch.rot90(probs, k=-rot90, dims=[-2, -1])
        if flip_v:
            probs = torch.flip(probs, dims=[-2])
        if flip_h:
            probs = torch.flip(probs, dims=[-1])
        return probs

    all_probs = []
    transforms = [
        (False, False, 0),
        (True, False, 0),
        (False, True, 0),
        (True, True, 0),
        (False, False, 1),
        (False, False, 2),
        (False, False, 3),
        (True, False, 1),
    ]

    with torch.no_grad():
        for flip_h, flip_v, rot90 in transforms:
            aug_kwargs = {**kwargs}
            img_transforms = get_image_kwargs(flip_h, flip_v, rot90)
            aug_kwargs.update(img_transforms)

            output = model(**aug_kwargs)
            probs = inverse_transform(output["probs"], flip_h, flip_v, rot90)
            all_probs.append(probs)

    all_probs_tensor = torch.stack(all_probs, dim=0)
    mean_probs = all_probs_tensor.mean(dim=0)
    std_probs = all_probs_tensor.std(dim=0)

    return {
        "probs": mean_probs,
        "std": std_probs,
        "all_probs": all_probs_tensor,
    }
