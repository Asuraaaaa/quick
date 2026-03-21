from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .backbones import TransformerBackbone, build_backbone
from .fusion import SingleLayerFusion, SingleLayerFusion_DCA, SingleLayerFusion_CA
from .mixstyle import MixStyle
from torch.autograd import Function

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


def grad_reverse(x, lambda_grl=1.0):
    return GradientReverseFunction.apply(x, lambda_grl)

class FaultDiagnosisNet(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]
        self.input_mode = data_cfg["input_mode"]
        self.backbone_name = model_cfg["backbone"]["name"]
        self.lambda_grl = model_cfg["lambda_grl"]
        dg_cfg = cfg.get("dg", {})
        self.dg_method = str(dg_cfg.get("method", "none")).lower()
        mix_cfg = dg_cfg.get("mixstyle", {})
        self.use_mixstyle = self.dg_method == "mixstyle" and bool(mix_cfg.get("enabled", False))
        self.mixstyle = MixStyle(
            p=float(mix_cfg.get("p", 0.5)),
            alpha=float(mix_cfg.get("alpha", 0.1)),
        )
        # project to common token dim=129 for transformer path
        if self.input_mode == "two_channel":
            self.input_proj = nn.Linear(63 * 2, 129)
        elif self.input_mode == "split_dual":
            self.input_proj = nn.Linear(63 * 2, 129)
        elif self.input_mode == "concat":
            self.input_proj = nn.Linear(126, 128)
        else:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")

        fusion_cfg = model_cfg["fusion"]
        if self.backbone_name == "transformer_dca": 
            self.fusion = SingleLayerFusion_DCA(
                d_model=fusion_cfg["d_model"],
                ffn_hidden=fusion_cfg["dim_feedforward"],
                n_head=fusion_cfg["nhead"],
                drop_prob=fusion_cfg["dropout"]
            )
        if self.backbone_name == "transformer_ca":
            self.fusion = SingleLayerFusion_CA(
                d_model=fusion_cfg["d_model"],
                ffn_hidden=fusion_cfg["dim_feedforward"],
                n_head=fusion_cfg["nhead"],
                drop_prob=fusion_cfg["dropout"]
            )
        if self.backbone_name == "transformer":
            self.fusion = SingleLayerFusion(
                d_model=fusion_cfg["d_model"],
                ffn_hidden=fusion_cfg["dim_feedforward"],
                n_head=fusion_cfg["nhead"],
                drop_prob=fusion_cfg["dropout"]
            )

        bb_cfg = model_cfg["backbone"]
        if self.backbone_name == "transformer" or self.backbone_name == "transformer_dca" or self.backbone_name == "transformer_ca":
            t_cfg = bb_cfg["transformer"]
            self.backbone = TransformerBackbone(
                d_model=t_cfg["d_model"],
                nhead=t_cfg["nhead"],
                dim_feedforward=t_cfg["dim_feedforward"],
                num_layers=t_cfg["num_layers"],
                dropout=bb_cfg["dropout"],
            )
            feat_dim = t_cfg["d_model"]
            self.transformer_head = nn.Sequential(nn.BatchNorm1d(feat_dim*129),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(feat_dim*129, feat_dim))
        elif self.backbone_name == "mobilenet_v2":
            self.backbone = build_backbone("mobilenet_v2", pretrained=False, out_dim=256)
            feat_dim = 256
        elif self.backbone_name == "mobilenet_v3_small":
            self.backbone = build_backbone("mobilenet_v3_small", pretrained=False, out_dim=256)
            feat_dim = 256
        elif self.backbone_name == "shufflenet_v2_x1_5":
            self.backbone = build_backbone("shufflenet_v2_x1_5", pretrained=False, out_dim=256)
            feat_dim = 256
        elif self.backbone_name == "efficientnet_b0":
            self.backbone = build_backbone("efficientnet_b0", pretrained=False, out_dim=256)
            feat_dim = 256
        elif self.backbone_name == "convnextv2_atto":
            self.backbone = build_backbone("convnextv2_atto", pretrained=False, out_dim=256)
            feat_dim = 256
        elif self.backbone_name == "mobilevit_xs":
            self.backbone = build_backbone('mobilevit_xs', pretrained=False, out_dim=256)
            feat_dim = 256
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        hidden1, hidden2 = model_cfg["classifier"]["hidden_dims"]
        drop = model_cfg["classifier"]["dropout"]
        
        self.classifier_head = nn.Sequential(
            nn.Linear(feat_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(drop),
            nn.Linear(hidden1, hidden2))
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(drop),
            nn.Linear(hidden2, data_cfg["num_classes"])
        )

        self.domain_classifier_head = nn.Sequential(
            nn.Linear(feat_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(drop),
            nn.Linear(hidden1, hidden2))
        self.domain_classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(drop),
            nn.Linear(hidden2, 6)
        )

        # self.phys_head_in_1 = nn.Linear(feat_dim, hidden2)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.phys_head_in_2 = nn.Linear(hidden2, 1)

        # self.phys_head_out_1 = nn.Linear(feat_dim, hidden2)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.phys_head_out_2 = nn.Linear(hidden2, 1)

    def forward(self, batch: Dict[str, torch.Tensor], lambda_grl=None) -> Dict[str, torch.Tensor]:
        if self.backbone_name == "transformer_dca":
            x1, x2 = batch["x1"], batch["x2"]
            tokens = self.fusion(x1, x2)
            feat = self.backbone(tokens)
            feat = feat.flatten(start_dim=1)
            feat = self.transformer_head(feat)
            if self.use_mixstyle:
                feat = self.mixstyle(feat)
            feat1 = self.classifier_head(feat)
            logits = self.classifier(feat1)
            feat_rev = grad_reverse(feat, self.lambda_grl)
            domain_feat1 = self.domain_classifier_head(feat_rev)
            domains = self.domain_classifier(domain_feat1)
            phi_in_feat = torch.tensor(0.0)
            phi_in_pred = torch.tensor(0.0)
            phi_out_feat = torch.tensor(0.0)
            phi_out_pred = torch.tensor(0.0)
            # phi_in_feat = self.phys_head_in_1(feat)
            # phi_in_pred = self.phys_head_in_2(self.relu1(phi_in_feat)).squeeze(-1)
            # phi_out_feat = self.phys_head_out_1(feat)
            # phi_out_pred = self.phys_head_out_2(self.relu1(phi_out_feat)).squeeze(-1)
        
        if self.backbone_name == "transformer_ca":
            x1, x2 = batch["x1"], batch["x2"]
            tokens = self.fusion(x1, x2)
            feat = self.backbone(tokens)
            feat = feat.flatten(start_dim=1)
            feat = self.transformer_head(feat)
            if self.use_mixstyle:
                feat = self.mixstyle(feat)
            feat1 = self.classifier_head(feat)
            logits = self.classifier(feat1)
            feat_rev = grad_reverse(feat, self.lambda_grl)
            domain_feat1 = self.domain_classifier_head(feat_rev)
            domains = self.domain_classifier(domain_feat1)
            phi_in_feat = torch.tensor(0.0)
            phi_in_pred = torch.tensor(0.0)
            phi_out_feat = torch.tensor(0.0)
            phi_out_pred = torch.tensor(0.0)
            # phi_in_feat = self.phys_head_in_1(feat)
            # phi_in_pred = self.phys_head_in_2(self.relu1(phi_in_feat)).squeeze(-1)
            # phi_out_feat = self.phys_head_out_1(feat)
            # phi_out_pred = self.phys_head_out_2(self.relu1(phi_out_feat)).squeeze(-1)

        if self.backbone_name == "transformer":
            x = batch["x"]
            x_in = self.input_proj(x)
            tokens = self.fusion(x_in)
            feat = self.backbone(tokens)
            feat = feat.flatten(start_dim=1)
            feat = self.transformer_head(feat)
            if self.use_mixstyle:
                feat = self.mixstyle(feat)
            feat1 = self.classifier_head(feat)
            logits = self.classifier(feat1)
            feat_rev = grad_reverse(feat, self.lambda_grl)
            domain_feat1 = self.domain_classifier_head(feat_rev)
            domains = self.domain_classifier(domain_feat1)
            phi_in_feat = torch.tensor(0.0)
            phi_in_pred = torch.tensor(0.0)
            phi_out_feat = torch.tensor(0.0)
            phi_out_pred = torch.tensor(0.0)
            # phi_in_feat = self.phys_head_in_1(feat)
            # phi_in_pred = self.phys_head_in_2(self.relu1(phi_in_feat)).squeeze(-1)
            # phi_out_feat = self.phys_head_out_1(feat)
            # phi_out_pred = self.phys_head_out_2(self.relu1(phi_out_feat)).squeeze(-1)
        
        if self.backbone_name == "mobilenet_v2":
            x = batch["x"]
            feat = self.backbone(x)
            if self.use_mixstyle:
                feat = self.mixstyle(feat)
            feat1 = self.classifier_head(feat)
            logits = self.classifier(feat1)
            feat_rev = grad_reverse(feat, self.lambda_grl)
            domain_feat1 = self.domain_classifier_head(feat_rev)
            domains = self.domain_classifier(domain_feat1)
            phi_in_feat = torch.tensor(0.0)
            phi_in_pred = torch.tensor(0.0)
            phi_out_feat = torch.tensor(0.0)
            phi_out_pred = torch.tensor(0.0)
            # phi_in_feat = self.phys_head_in_1(feat)
            # phi_in_pred = self.phys_head_in_2(self.relu1(phi_in_feat)).squeeze(-1)
            # phi_out_feat = self.phys_head_out_1(feat)
            # phi_out_pred = self.phys_head_out_2(self.relu1(phi_out_feat)).squeeze(-1)
        
        if self.backbone_name == "mobilenet_v3_small":
            x = batch["x"]
            feat = self.backbone(x)
            if self.use_mixstyle:
                feat = self.mixstyle(feat)
            feat1 = self.classifier_head(feat)
            logits = self.classifier(feat1)
            feat_rev = grad_reverse(feat, self.lambda_grl)
            domain_feat1 = self.domain_classifier_head(feat_rev)
            domains = self.domain_classifier(domain_feat1)
            phi_in_feat = torch.tensor(0.0)
            phi_in_pred = torch.tensor(0.0)
            phi_out_feat = torch.tensor(0.0)
            phi_out_pred = torch.tensor(0.0)
            # phi_in_feat = self.phys_head_in_1(feat)
            # phi_in_pred = self.phys_head_in_2(self.relu1(phi_in_feat)).squeeze(-1)
            # phi_out_feat = self.phys_head_out_1(feat)
            # phi_out_pred = self.phys_head_out_2(self.relu1(phi_out_feat)).squeeze(-1)
        
        if self.backbone_name == "shufflenet_v2_x1_5":
            x = batch["x"]
            feat = self.backbone(x)
            if self.use_mixstyle:
                feat = self.mixstyle(feat)
            feat1 = self.classifier_head(feat)
            logits = self.classifier(feat1)
            feat_rev = grad_reverse(feat, self.lambda_grl)
            domain_feat1 = self.domain_classifier_head(feat_rev)
            domains = self.domain_classifier(domain_feat1)
            phi_in_feat = torch.tensor(0.0)
            phi_in_pred = torch.tensor(0.0)
            phi_out_feat = torch.tensor(0.0)
            phi_out_pred = torch.tensor(0.0)
            # phi_in_feat = self.phys_head_in_1(feat)
            # phi_in_pred = self.phys_head_in_2(self.relu1(phi_in_feat)).squeeze(-1)
            # phi_out_feat = self.phys_head_out_1(feat)
            # phi_out_pred = self.phys_head_out_2(self.relu1(phi_out_feat)).squeeze(-1)
        
        if self.backbone_name == "efficientnet_b0":
            x = batch["x"]
            feat = self.backbone(x)
            if self.use_mixstyle:
                feat = self.mixstyle(feat)
            feat1 = self.classifier_head(feat)
            logits = self.classifier(feat1)
            feat_rev = grad_reverse(feat, self.lambda_grl)
            domain_feat1 = self.domain_classifier_head(feat_rev)
            domains = self.domain_classifier(domain_feat1)
            phi_in_feat = torch.tensor(0.0)
            phi_in_pred = torch.tensor(0.0)
            phi_out_feat = torch.tensor(0.0)
            phi_out_pred = torch.tensor(0.0)
            # phi_in_feat = self.phys_head_in_1(feat)
            # phi_in_pred = self.phys_head_in_2(self.relu1(phi_in_feat)).squeeze(-1)
            # phi_out_feat = self.phys_head_out_1(feat)
            # phi_out_pred = self.phys_head_out_2(self.relu1(phi_out_feat)).squeeze(-1)
        
        if self.backbone_name == "convnextv2_atto":
            x = batch["x"]
            feat = self.backbone(x)
            if self.use_mixstyle:
                feat = self.mixstyle(feat)
            feat1 = self.classifier_head(feat)
            logits = self.classifier(feat1)
            feat_rev = grad_reverse(feat, self.lambda_grl)
            domain_feat1 = self.domain_classifier_head(feat_rev)
            domains = self.domain_classifier(domain_feat1)
            phi_in_feat = torch.tensor(0.0)
            phi_in_pred = torch.tensor(0.0)
            phi_out_feat = torch.tensor(0.0)
            phi_out_pred = torch.tensor(0.0)
            # phi_in_feat = self.phys_head_in_1(feat)
            # phi_in_pred = self.phys_head_in_2(self.relu1(phi_in_feat)).squeeze(-1)
            # phi_out_feat = self.phys_head_out_1(feat)
            # phi_out_pred = self.phys_head_out_2(self.relu1(phi_out_feat)).squeeze(-1)
            
        if self.backbone_name == "mobilevit_xs":
            x = batch["x"]
            feat = self.backbone(x)
            if self.use_mixstyle:
                feat = self.mixstyle(feat)
            feat1 = self.classifier_head(feat)
            logits = self.classifier(feat1)
            feat_rev = grad_reverse(feat, self.lambda_grl)
            domain_feat1 = self.domain_classifier_head(feat_rev)
            domains = self.domain_classifier(domain_feat1)
            phi_in_feat = torch.tensor(0.0)
            phi_in_pred = torch.tensor(0.0)
            phi_out_feat = torch.tensor(0.0)
            phi_out_pred = torch.tensor(0.0)
            # phi_in_feat = self.phys_head_in_1(feat)
            # phi_in_pred = self.phys_head_in_2(self.relu1(phi_in_feat)).squeeze(-1)
            # phi_out_feat = self.phys_head_out_1(feat)
            # phi_out_pred = self.phys_head_out_2(self.relu1(phi_out_feat)).squeeze(-1)

        return {
            "feat": feat,
            "logits": logits,
            "domains": domains,
            "phi_in_feat": phi_in_feat,
            "phi_in_pred": phi_in_pred,
            "phi_out_feat": phi_out_feat,
            "phi_out_pred": phi_out_pred
        }
