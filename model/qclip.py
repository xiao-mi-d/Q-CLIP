from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn

import model.core.vision_encoder.pe as pe
import model.core.vision_encoder.transforms as pe_transforms


DEFAULT_QUALITY_PROMPTS = (
    "X X X a video of bad quality",
    "X X X a video of poor quality",
    "X X X a video of fair quality",
    "X X X a video of good quality",
    "X X X a video of excellent quality",
)


class QCLIP(nn.Module):
    """Q-CLIP model with shared cross-modal adapters and five quality prompts."""

    def __init__(
        self,
        pe_config: str = "PE-Core-L14-336",
        checkpoint_path: str | None = "./pretrained_weights/PE-Core-L14-336.pt",
        pretrained: bool = True,
        shared_dim: int = 128,
        adapter_start_layer: int = 18,
        num_context_tokens: int = 3,
        quality_prompts: Sequence[str] = DEFAULT_QUALITY_PROMPTS,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        if checkpoint_path is not None:
            checkpoint_path = os.path.expanduser(os.path.expandvars(checkpoint_path))
            checkpoint = Path(checkpoint_path)
            checkpoint_path = str(checkpoint) if checkpoint.exists() else checkpoint_path

        self.clip_model = pe.CLIP.from_config(
            pe_config,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
        )
        self.vit = self.clip_model.visual
        self.text_encoder = self.clip_model.transformer
        self.adapter_start_layer = adapter_start_layer
        self.num_context_tokens = num_context_tokens

        self.vit_dim = self.vit.conv1.out_channels
        self.text_dim = self.text_encoder.width
        if self.vit_dim != self.text_dim:
            raise ValueError(
                "Q-CLIP shares adapter projections across visual and text branches, "
                f"but got visual dim {self.vit_dim} and text dim {self.text_dim}."
            )

        if freeze_backbone:
            for parameter in self.clip_model.parameters():
                parameter.requires_grad = False

        self.patch_size = self.vit.patch_size
        self.width = self.vit_dim

        self.shared_core = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, shared_dim),
        )
        self.encoder_adapters = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "down_proj": nn.Linear(self.vit_dim, shared_dim),
                        "up_proj": nn.Linear(shared_dim, self.vit_dim),
                        "ln": nn.LayerNorm(self.vit_dim),
                    }
                )
                for _ in range(len(self.vit.transformer.resblocks))
            ]
        )
        self.proj_adapter = nn.Sequential(
            nn.Linear(self.vit_dim, shared_dim),
            nn.Linear(shared_dim, shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, self.vit_dim),
        )

        self.image_adapter_weight = nn.Parameter(torch.tensor(0.1))
        self.text_adapter_weight = nn.Parameter(torch.tensor(0.1))
        self.image_proj_adapter_weight = nn.Parameter(torch.tensor(0.1))
        self.text_proj_adapter_weight = nn.Parameter(torch.tensor(0.1))

        self.tokenizer = pe_transforms.get_text_tokenizer(self.clip_model.context_length)
        text_tokens = self.tokenizer(list(quality_prompts))
        self.register_buffer("text_tokens", text_tokens, persistent=False)

        with torch.no_grad():
            text_embeddings = self.clip_model.token_embedding(text_tokens)

        self.context_tokens = nn.Parameter(
            text_embeddings[0:1, 1 : 1 + num_context_tokens].clone()
        )
        self.register_buffer("prompt_prefix", text_embeddings[:, :1, :].clone())
        self.register_buffer(
            "prompt_suffix",
            text_embeddings[:, 1 + num_context_tokens :, :].clone(),
        )

        self.score_proj = nn.Parameter(torch.empty(len(quality_prompts), 1))
        nn.init.normal_(self.score_proj, std=len(quality_prompts) ** -0.5)

    def build_prompts(self) -> torch.Tensor:
        return torch.cat(
            [
                self.prompt_prefix,
                self.context_tokens.repeat(self.prompt_prefix.shape[0], 1, 1),
                self.prompt_suffix,
            ],
            dim=1,
        )

    def _apply_encoder_adapter(
        self,
        x: torch.Tensor,
        adapter: nn.ModuleDict,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        adapter_out = adapter["down_proj"](x)
        adapter_out = self.shared_core(adapter_out)
        adapter_out = adapter["up_proj"](adapter_out)
        adapter_out = adapter["ln"](adapter_out)
        return x + scale * adapter_out

    def forward_image(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size

        x = self.clip_model.visual.conv1(x)
        x = x.permute(0, 2, 3, 1).reshape(batch, -1, self.width)
        x = torch.cat(
            [
                self.clip_model.visual.class_embedding.view(1, 1, -1).expand(
                    batch, -1, -1
                ),
                x,
            ],
            dim=1,
        )

        x = x + self.clip_model.visual._sample_abs_posemb(grid_h, grid_w)
        self.clip_model.visual.rope.update_grid(x.device, grid_h, grid_w)
        x = self.clip_model.visual.ln_pre(x)

        for i, block in enumerate(self.vit.transformer.resblocks):
            x = block(x)
            if i >= self.adapter_start_layer:
                x = self._apply_encoder_adapter(
                    x,
                    self.encoder_adapters[i],
                    self.image_adapter_weight,
                )

        x = self.clip_model.visual.ln_post(x)
        x = self.clip_model.visual.attn_pool(x).squeeze(1)
        adapter_feature = self.proj_adapter(x)
        x = x @ self.clip_model.visual.proj
        return x + self.image_proj_adapter_weight * adapter_feature

    def forward_video(self, video: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = video.shape
        video = video.reshape(batch * frames, channels, height, width)
        frame_features = self.forward_image(video)
        frame_features = frame_features.reshape(batch, frames, -1)
        return frame_features.mean(dim=1)

    def forward_text(self, prompts: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        seq_len = text.shape[1]
        attn_mask = self.clip_model.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]

        x = prompts + self.clip_model.positional_embedding[:seq_len]
        for i, block in enumerate(self.text_encoder.resblocks):
            x = block(x, attn_mask=attn_mask)
            if i >= self.adapter_start_layer:
                x = self._apply_encoder_adapter(
                    x,
                    self.encoder_adapters[i],
                    self.text_adapter_weight,
                )

        x = self.clip_model.ln_final(x)
        pooled, _ = self.clip_model.text_global_pool(
            x,
            text,
            pool_type=self.clip_model.pool_type,
        )
        adapter_feature = self.proj_adapter(pooled)
        pooled = pooled @ self.clip_model.text_projection
        return pooled + self.text_proj_adapter_weight * adapter_feature

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        prompts = self.build_prompts()
        text_features = self.forward_text(prompts, self.text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        video_features = self.forward_video(video)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        similarity = video_features @ text_features.T
        score = similarity @ self.score_proj
        return score.view(-1)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)


CLIP_VQA = QCLIP
