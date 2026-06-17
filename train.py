from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import kendalltau, pearsonr, spearmanr
from tqdm import tqdm

import datasets
from model import QCLIP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rank_loss(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if y_pred.shape[0] < 2:
        return y_pred.new_tensor(0.0)
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()


def plcc_loss(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sigma_hat, mean_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - mean_hat) / (sigma_hat + 1e-8)
    sigma, mean = torch.std_mean(y, unbiased=False)
    y = (y - mean) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()


def rescale(pred: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    pred_std = np.std(pred)
    if pred_std < 1e-8:
        return np.full_like(pred, np.mean(target) if target is not None else 0.0)
    pred = (pred - np.mean(pred)) / pred_std
    if target is None:
        return pred
    return pred * np.std(target) + np.mean(target)


def build_loaders(opt: dict[str, Any], device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    train_loaders = {}
    val_loaders = {}
    pin_memory = device.startswith("cuda")

    for key, data_cfg in opt["data"].items():
        dataset = datasets.build_dataset(data_cfg["type"], data_cfg["args"])
        if key.startswith("train"):
            train_loaders[key] = torch.utils.data.DataLoader(
                dataset,
                batch_size=opt["batch_size"],
                num_workers=opt["num_workers"],
                shuffle=True,
                pin_memory=pin_memory,
            )
        elif key.startswith("val"):
            val_loaders[key] = torch.utils.data.DataLoader(
                dataset,
                batch_size=opt.get("val_batch_size", 1),
                num_workers=opt["num_workers"],
                shuffle=False,
                pin_memory=pin_memory,
            )
    return train_loaders, val_loaders


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str) -> None:
    checkpoint_path = os.path.expanduser(os.path.expandvars(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        key.replace("module.", "", 1): value for key, value in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")


def train_epoch(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: str,
    epoch: int,
) -> None:
    model.train()
    for data in tqdm(loader, desc=f"Training epoch {epoch}"):
        optimizer.zero_grad(set_to_none=True)
        video = data["video"].to(device, non_blocking=True)
        target = data["gt_label"].float().to(device, non_blocking=True).view(-1, 1)

        pred = model(video).view(-1, 1)
        loss = plcc_loss(pred, target) + 0.3 * rank_loss(pred, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
    model.eval()


def evaluate(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: str,
    best: tuple[float, float, float, float],
    save_model: bool,
    save_path: Path,
    suffix: str,
) -> tuple[float, float, float, float]:
    model.eval()
    predictions = []
    targets = []

    for data in tqdm(loader, desc=f"Validating {suffix}"):
        video = data["video"].to(device, non_blocking=True)
        target = data["gt_label"].float().cpu().numpy().reshape(-1)
        with torch.no_grad():
            pred = model(video).detach().cpu().numpy().reshape(-1)
        targets.extend(target.tolist())
        predictions.extend(pred.tolist())

    targets_np = np.asarray(targets, dtype=np.float64)
    predictions_np = rescale(np.asarray(predictions, dtype=np.float64), targets_np)

    srocc = float(spearmanr(targets_np, predictions_np)[0])
    plcc = float(pearsonr(targets_np, predictions_np)[0])
    krocc = float(kendalltau(targets_np, predictions_np)[0])
    rmse = float(np.sqrt(np.mean((targets_np - predictions_np) ** 2)))

    best_s, best_p, best_k, best_r = best
    if save_model and srocc + plcc > best_s + best_p:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "validation_results": {
                    "srocc": srocc,
                    "plcc": plcc,
                    "krocc": krocc,
                    "rmse": rmse,
                },
            },
            save_path,
        )

    best = (
        max(best_s, srocc),
        max(best_p, plcc),
        max(best_k, krocc),
        min(best_r, rmse),
    )

    print(
        f"[{suffix}] SROCC {srocc:.4f} best {best[0]:.4f} | "
        f"PLCC {plcc:.4f} best {best[1]:.4f} | "
        f"KROCC {krocc:.4f} best {best[2]:.4f} | "
        f"RMSE {rmse:.4f} best {best[3]:.4f}"
    )
    return best


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    opt: dict[str, Any],
    train_loaders: dict[str, torch.utils.data.DataLoader],
) -> torch.optim.lr_scheduler.LambdaLR:
    steps_per_epoch = max(1, sum(len(loader) for loader in train_loaders.values()))
    warmup_iter = int(opt["warmup_epochs"] * steps_per_epoch)
    total_iter = max(1, int(opt["num_epochs"] * steps_per_epoch))

    def lr_lambda(cur_iter: int) -> float:
        if warmup_iter > 0 and cur_iter <= warmup_iter:
            return cur_iter / warmup_iter
        progress = (cur_iter - warmup_iter) / max(1, total_iter - warmup_iter)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train(
    train_loaders: dict[str, torch.utils.data.DataLoader],
    val_loaders: dict[str, torch.utils.data.DataLoader],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    opt: dict[str, Any],
    device: str,
    bests: dict[str, tuple[float, float, float, float]],
) -> None:
    output_dir = Path(opt.get("output_dir", "pretrained_weights"))
    for epoch in range(int(opt["num_epochs"])):
        print(f"Epoch {epoch}")
        for loader in train_loaders.values():
            train_epoch(loader, model, optimizer, scheduler, device, epoch)

        for key, loader in val_loaders.items():
            save_path = output_dir / f"{opt['name']}_{key}_best.pth"
            bests[key] = evaluate(
                loader,
                model,
                device,
                bests[key],
                save_model=bool(opt.get("save_model", False)),
                save_path=save_path,
                suffix=key,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or fine-tune Q-CLIP.")
    parser.add_argument(
        "-o",
        "--opt",
        default="options/pretrain/qclip_lsvq.yml",
        help="Path to a YAML option file.",
    )
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0.")
    args = parser.parse_args()

    with open(args.opt, "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)

    set_seed(int(opt.get("seed", 42)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_args = dict(opt.get("model", {}))
    if "pe_checkpoint" in opt:
        model_args.setdefault("checkpoint_path", opt["pe_checkpoint"])
    model = QCLIP(**model_args)
    if opt.get("load_path"):
        load_checkpoint(model, opt["load_path"], device)
    model = model.to(device)

    train_loaders, val_loaders = build_loaders(opt, device)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(opt["optimizer"]["lr"]),
        weight_decay=float(opt["optimizer"]["wd"]),
    )
    scheduler = build_scheduler(optimizer, opt, train_loaders)
    bests = {key: (-1.0, -1.0, -1.0, 1000.0) for key in val_loaders}

    train(
        train_loaders,
        val_loaders,
        model,
        optimizer,
        scheduler,
        opt,
        device,
        bests,
    )


if __name__ == "__main__":
    main()
