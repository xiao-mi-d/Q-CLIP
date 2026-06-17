from __future__ import annotations

import glob
import os
import os.path as osp
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

try:
    import decord
except ModuleNotFoundError:  # pragma: no cover - only hit in incomplete envs.
    decord = None


@dataclass
class VideoRecord:
    filename: str
    label: float


def _read_annotations(
    anno_file: str,
    data_prefix: str,
    strip_suffix: bool = False,
) -> list[VideoRecord]:
    anno_file = osp.expanduser(os.path.expandvars(anno_file))
    data_prefix = osp.expanduser(os.path.expandvars(data_prefix))
    records: list[VideoRecord] = []

    with open(anno_file, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            fields = [field.strip() for field in line.strip().split(",")]
            if len(fields) != 4:
                raise ValueError(
                    f"{anno_file}:{line_no} must use the 4-column format: "
                    "filename,duration,fps,label."
                )

            filename = fields[0]
            if strip_suffix:
                filename = osp.splitext(filename)[0]
            records.append(
                VideoRecord(
                    filename=osp.join(data_prefix, filename),
                    label=float(fields[3]),
                )
            )
    return records


def _apply_split(records: list[VideoRecord], split: dict[str, Any] | None) -> list[VideoRecord]:
    if not split:
        return records

    ratio = float(split.get("ratio", 0.8))
    role = split.get("role", "train")
    seed = int(split.get("seed", 42))
    if role not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split role: {role}")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * ratio)
    return shuffled[:cut] if role == "train" else shuffled[cut:]


def _build_image_transform(
    image_size: int = 336,
    center_crop: bool = False,
    interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
) -> T.Compose:
    if center_crop:
        crop = [T.Resize(image_size, interpolation=interpolation), T.CenterCrop(image_size)]
    else:
        crop = [T.Resize((image_size, image_size), interpolation=interpolation)]

    return T.Compose(
        crop
        + [
            T.Lambda(lambda image: image.convert("RGB")),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
        ]
    )


def _frame_indices(total_frames: int, num_frames: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("Video has no readable frames.")
    indices = np.linspace(0, total_frames - 1, num_frames).round().astype(int).tolist()
    return [max(0, min(total_frames - 1, index)) for index in indices]


class QCLIPVideoDataset(torch.utils.data.Dataset):
    """Video MOS dataset."""

    def __init__(self, opt: dict[str, Any]):
        super().__init__()
        self.opt = dict(opt)
        self.ann_file = self.opt["anno_file"]
        self.data_prefix = self.opt.get("data_prefix", "")
        self.phase = self.opt.get("phase", "train")
        self.num_frames = int(self.opt.get("num_frames", 8))
        self.image_size = int(self.opt.get("image_size", 336))
        self.center_crop = bool(self.opt.get("center_crop", False))
        self.strip_suffix = bool(self.opt.get("strip_suffix", False))
        self.transform = _build_image_transform(
            image_size=self.image_size,
            center_crop=self.center_crop,
        )

        records = _read_annotations(
            self.ann_file,
            self.data_prefix,
            strip_suffix=self.strip_suffix,
        )
        self.video_infos = _apply_split(records, self.opt.get("split"))

    def __len__(self) -> int:
        return len(self.video_infos)

    def preprocess_video(self, record: VideoRecord) -> torch.Tensor:
        if decord is None:
            raise ImportError("decord is required for video loading. Install requirements.txt.")

        vr = decord.VideoReader(record.filename)
        frame_indices = _frame_indices(len(vr), self.num_frames)
        frames = vr.get_batch(frame_indices)
        if torch.is_tensor(frames):
            frames = frames.cpu().numpy()
        elif hasattr(frames, "asnumpy"):
            frames = frames.asnumpy()

        frames = [self.transform(Image.fromarray(frame)) for frame in frames]
        return torch.stack(frames, dim=0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.video_infos[index]
        return {
            "video": self.preprocess_video(record),
            "gt_label": record.label,
            "name": osp.basename(record.filename),
        }


class QCLIPFrameFolderDataset(QCLIPVideoDataset):
    """Dataset for pre-extracted frame folders, e.g. LIVE-Qualcomm."""

    def preprocess_video(self, record: VideoRecord) -> torch.Tensor:
        image_files = sorted(
            glob.glob(osp.join(record.filename, "*.png"))
            + glob.glob(osp.join(record.filename, "*.jpg"))
            + glob.glob(osp.join(record.filename, "*.jpeg"))
        )
        if not image_files:
            raise FileNotFoundError(f"No frame images found under {record.filename}")

        frame_indices = _frame_indices(len(image_files), self.num_frames)
        frames = []
        for index in frame_indices:
            with Image.open(image_files[index]) as image:
                frames.append(self.transform(image))
        return torch.stack(frames, dim=0)


DATASET_REGISTRY = {
    "QCLIPVideoDataset": QCLIPVideoDataset,
    "QCLIPFrameFolderDataset": QCLIPFrameFolderDataset,
}


def build_dataset(dataset_type: str, args: dict[str, Any]) -> torch.utils.data.Dataset:
    if dataset_type not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset type '{dataset_type}'. Available: {available}")
    return DATASET_REGISTRY[dataset_type](args)
