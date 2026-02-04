# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import warnings
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from loguru import logger


class Dino(torch.nn.Module):
    supports_packed_batch: ClassVar[bool] = True

    def __init__(
        self,
        input_size: int = 224,
        repo_or_dir: str = 'facebookresearch/dinov2',
        dino_model: str = 'dinov2_vitb14',
        source: str = 'github',
        backbone_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        # for backward compatible
        prenorm_features: bool = False,
        freeze_backbone: bool = True,
        prune_network: bool = False,  # False for backward compatible
    ):
        super().__init__()
        if backbone_kwargs is None:
            backbone_kwargs = {}

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            logger.info(f'Loading DINO model: {dino_model} from {repo_or_dir} (source: {source})')
            if backbone_kwargs:
                logger.info(f'DINO backbone kwargs: {backbone_kwargs}')
            self.backbone = torch.hub.load(
                repo_or_dir=repo_or_dir,
                model=dino_model,
                source=source,
                verbose=False,
                **backbone_kwargs,
            )
            logger.info(
                f'Loaded DINO model - type: {type(self.backbone)}, '
                f'embed_dim: {self.backbone.embed_dim}, '
                f'patch_size: {getattr(self.backbone.patch_embed, "patch_size", "N/A")}'
            )

        self.resize_input_size = (input_size, input_size)
        self.embed_dim = self.backbone.embed_dim
        self.input_size = input_size
        self.input_channels = 3
        self.normalize_images = normalize_images
        self.prenorm_features = prenorm_features
        self.register_buffer(
            'mean', torch.as_tensor([[0.485, 0.456, 0.406]]).view(-1, 1, 1), persistent=False
        )
        self.register_buffer(
            'std', torch.as_tensor([[0.229, 0.224, 0.225]]).view(-1, 1, 1), persistent=False
        )

        if prune_network:
            self._prune_network()

        # freeze
        if freeze_backbone:
            self.requires_grad_(False)
            self.eval()
        elif not prune_network:
            logger.warning(
                'Unfreeze encoder w/o prune parameter may lead to error in ddp/fp16 training'
            )

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        resized = torch.nn.functional.interpolate(
            x,
            size=self.resize_input_size,
            mode='bilinear',
            align_corners=False,
        )

        if resized.shape[1] == 1:
            resized = resized.repeat(1, 3, 1, 1)

        if self.normalize_images:
            resized = resized.sub_(self.mean).div_(self.std)
        return resized

    def _forward_last_layer(self, input_img: torch.Tensor) -> torch.Tensor:
        output = self.backbone.forward_features(input_img)
        if self.prenorm_features:
            features = output['x_prenorm']
            return F.layer_norm(features, features.shape[-1:])

        return torch.cat(
            [
                output['x_norm_clstoken'].unsqueeze(1),
                output['x_norm_patchtokens'],
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...], **kwargs):
        _ = kwargs

        if isinstance(x, (list, tuple)):
            assert len(x) > 0, 'Expected non-empty list/tuple of inputs'
            assert all(isinstance(v, torch.Tensor) for v in x), 'All packed inputs must be tensors'
            assert all(v.dim() == 4 for v in x), 'Packed inputs must be [B,C,H,W] tensors'
            assert all(v.dtype == x[0].dtype for v in x), 'Packed inputs must share dtype'
            assert all(v.device == x[0].device for v in x), 'Packed inputs must share device'

            batch_sizes = [int(v.shape[0]) for v in x]
            pre = [self._preprocess_input(v) for v in x]
            packed = torch.cat(pre, dim=0)
            tokens_packed = self._forward_last_layer(packed)
            tokens_packed = tokens_packed.to(x[0].dtype)
            tokens_list = list(torch.split(tokens_packed, batch_sizes, dim=0))
            return tokens_list

        tokens = self._forward_last_layer(self._preprocess_input(x))
        return tokens.to(x.dtype)

    def _prune_network(self) -> None:
        self.backbone.mask_token = None
        if self.prenorm_features:
            self.backbone.norm = torch.nn.Identity()


class DinoForMasks(torch.nn.Module):
    def __init__(
        self,
        backbone: Dino,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self.backbone.embed_dim

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward(mask)
