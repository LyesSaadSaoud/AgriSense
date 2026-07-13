from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel


class HFVisionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        out_dim: int = 256,
        freeze_backbone: bool = True,
        dropout: float = 0.1,
        pooling: str = "cls",
    ):
        super().__init__()
        self.pooling = pooling
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        last_hidden = outputs.last_hidden_state  # [B, N, H]

        if self.pooling == "cls":
            pooled = last_hidden[:, 0]  # CLS token
        elif self.pooling == "mean":
            pooled = last_hidden.mean(dim=1)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")

        return self.proj(pooled)


class HFTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        out_dim: int = 128,
        freeze_backbone: bool = True,
        dropout: float = 0.1,
        pooling: str = "cls",
    ):
        super().__init__()
        self.pooling = pooling
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, T, H]

        if self.pooling == "cls":
            pooled = last_hidden[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / denom
        else:
            raise ValueError(f"Unsupported text pooling mode: {self.pooling}")

        return self.proj(pooled)


class SensorEncoder(nn.Module):
    def __init__(self, input_dim: int, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GreenhouseVLA(nn.Module):
    def __init__(
        self,
        sensor_dim: int = 19,
        image_model_name: str = "facebook/dinov2-small",
        text_model_name: str = "bert-base-uncased",
        image_dim: int = 256,
        text_dim: int = 128,
        sensor_hidden_dim: int = 128,
        fusion_dim: int = 256,
        num_zone_classes: int = 4,
        regression_dim: int = 12,
        freeze_image_backbone: bool = True,
        freeze_text_backbone: bool = True,
        image_pooling: str = "cls",
        text_pooling: str = "cls",
    ):
        super().__init__()

        self.image_encoder = HFVisionEncoder(
            model_name=image_model_name,
            out_dim=image_dim,
            freeze_backbone=freeze_image_backbone,
            dropout=0.1,
            pooling=image_pooling,
        )

        self.text_encoder = HFTextEncoder(
            model_name=text_model_name,
            out_dim=text_dim,
            freeze_backbone=freeze_text_backbone,
            dropout=0.1,
            pooling=text_pooling,
        )

        self.sensor_encoder = SensorEncoder(
            input_dim=sensor_dim,
            out_dim=sensor_hidden_dim,
            dropout=0.1,
        )

        fused_in = image_dim + text_dim + sensor_hidden_dim

        self.fusion = nn.Sequential(
            nn.Linear(fused_in, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, regression_dim),
        )

        self.zone_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, num_zone_classes),
        )

        # Binary pesticide prediction head
        self.pesticide_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, 1),
        )

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sensor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        img_feat = self.image_encoder(image)
        txt_feat = self.text_encoder(input_ids, attention_mask)
        sensor_feat = self.sensor_encoder(sensor)

        fused = torch.cat([img_feat, txt_feat, sensor_feat], dim=-1)
        fused = self.fusion(fused)

        pred_reg = self.regression_head(fused)
        pred_zone_logits = self.zone_head(fused)
        pred_pesticide_logits = self.pesticide_head(fused)
        pred_pesticide_prob = torch.sigmoid(pred_pesticide_logits)

        return {
            "pred_regression": pred_reg,
            "pred_zone_logits": pred_zone_logits,
            "pred_pesticide_logits": pred_pesticide_logits,
            "pred_pesticide_prob": pred_pesticide_prob,
            "fused": fused,
            "img_feat": img_feat,
            "txt_feat": txt_feat,
            "sensor_feat": sensor_feat,
        }
