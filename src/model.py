"""This file contains functions to train the model."""

import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset / sampler helpers
# ---------------------------------------------------------------------------

class TensorListDataset(Dataset):
    def __init__(self, X_list, y_array, transform=None):
        self.X_list = X_list
        self.y_array = y_array
        self.transform = transform

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        x = self.X_list[idx].clone()
        if self.transform is not None:
            x = self.transform(x)
        y = torch.tensor(self.y_array[idx], dtype=torch.float32)
        return x, y


def weighted_sampler_from_y(y, seed=None):
    y_int = np.rint(y).astype(int)
    unique, counts = np.unique(y_int, return_counts=True)
    freq = {k: c for k, c in zip(unique, counts)}
    weights = np.array([1.0 / freq[v] for v in y_int], dtype=np.float32)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return WeightedRandomSampler(
        torch.tensor(weights),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def prepare_tensors(data, img_size=(64, 64)):
    X, y, groups = [], [], []
    for d in data:
        img = d["image"].astype(np.float32) / 255.0
        if img.ndim != 2:
            raise ValueError(f"Expected grayscale 2D image, got shape {img.shape}")
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=img_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        X.append(img)
        y.append(float(d["LevelingScore"]))
        groups.append(d.get("PanelID", None))
    return X, np.array(y, dtype=np.float32), np.array(groups)


def make_transforms(img_size, rotation_deg, crop_scale):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(rotation_deg),
        transforms.RandomResizedCrop(img_size, scale=(crop_scale, 1.0), antialias=True),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    val_tf = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return train_tf, val_tf


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class WaveletCNNRegressor(nn.Module):
    def __init__(self, num_blocks=5, base_channels=16, hidden_dim=128, dropout=0.5, use_batchnorm=True):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        layers = []
        in_channels = 1
        channels = base_channels
        for i in range(num_blocks):
            layers.append(nn.Conv2d(in_channels, channels, 3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            layers.extend([
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(),
            ])
            if i < num_blocks - 1:
                layers.append(nn.MaxPool2d(2))
            in_channels = channels
            channels *= 2
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)
        self.regressor = nn.Sequential(
            nn.Linear(in_channels + 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x_low1 = self.pool(x)
        x_low2 = self.pool(x_low1)
        m0 = x.mean(dim=[2, 3])
        m1 = x_low1.mean(dim=[2, 3])
        m2 = x_low2.mean(dim=[2, 3])
        wavelet_feats = torch.cat([m0, m1, m2], dim=1)
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        combined = torch.cat([feats, wavelet_feats], dim=1)
        return self.regressor(combined).squeeze(1)


def make_optimizer(name, params, lr, weight_decay):
    if name == "Adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "RMSprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)


def make_loss(name):
    if name == "L1":
        return nn.L1Loss()
    if name == "MSE":
        return nn.MSELoss()
    return nn.SmoothL1Loss(beta=0.5)


# ---------------------------------------------------------------------------
# Hyperparameters (best found via Optuna)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "img_size": 92,
    "num_blocks": 5,
    "base_channels": 12,
    "hidden_dim": 256,
    "dropout": 0.2511572371061875,
    "use_batchnorm": True,
    "batch_size": 32,
    "optimizer": "Adam",
    "lr": 0.0006049716507542575,
    "weight_decay": 1.963434157293331e-08,
    "rotation_deg": 6,
    "crop_scale": 0.9608106745617722,
    "loss": "SmoothL1",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_and_evaluate(
    train_data,
    test_data,
    params=None,
    epochs=70,
    run_seed=3,
    save_dir="../models",
    device=None,
):
    """
    Train WaveletCNNRegressor on train_data and evaluate on test_data.

    Returns
    -------
    model : trained WaveletCNNRegressor
    results_df : DataFrame with PanelID, Prediction, TrueLabel, Difference, AbsDifference
    preds_np, labels_np, groups : raw arrays for downstream analysis
    """
    if params is None:
        params = DEFAULT_PARAMS
    if device is None:
        device = DEVICE

    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    img_size = int(params["img_size"])
    X_std, y_std, _ = prepare_tensors(train_data, img_size=(img_size, img_size))
    X_real, y_real, groups_real = prepare_tensors(test_data, img_size=(img_size, img_size))

    train_tf, _ = make_transforms(
        img_size=img_size,
        rotation_deg=int(params["rotation_deg"]),
        crop_scale=float(params["crop_scale"]),
    )

    std_train_ds = TensorListDataset(X_std, y_std, transform=train_tf)
    std_sampler = weighted_sampler_from_y(y_std, seed=run_seed)
    std_train_loader = DataLoader(
        std_train_ds,
        batch_size=int(params["batch_size"]),
        sampler=std_sampler,
        drop_last=False,
    )

    val_tf = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])
    real_test_ds = TensorListDataset(X_real, y_real, transform=val_tf)
    real_test_loader = DataLoader(
        real_test_ds, batch_size=int(params["batch_size"]), shuffle=False, drop_last=False
    )

    model = WaveletCNNRegressor(
        num_blocks=int(params["num_blocks"]),
        base_channels=int(params["base_channels"]),
        hidden_dim=int(params["hidden_dim"]),
        dropout=float(params["dropout"]),
        use_batchnorm=bool(params["use_batchnorm"]),
    ).to(device)

    optimizer = make_optimizer(
        params["optimizer"], model.parameters(), float(params["lr"]), float(params["weight_decay"])
    )
    criterion = make_loss(params["loss"])

    print("=" * 76)
    print("WaveletCNNRegressor | train: standards | eval: real paint")
    print(f"Seed: {run_seed}  |  Train: {len(X_std)}  |  Test: {len(X_real)}")
    print("=" * 76)

    for epoch in range(epochs):
        model.train()
        running_loss, n_seen = 0.0, 0
        for images, labels in std_train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            n_seen += images.size(0)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs} - train loss: {running_loss / max(1, n_seen):.4f}")

    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for images, labels in real_test_loader:
            preds_all.append(model(images.to(device)).detach().cpu().numpy())
            labels_all.append(labels.numpy())

    preds_np = np.concatenate(preds_all)
    labels_np = np.concatenate(labels_all)
    abs_err = np.abs(preds_np - labels_np)

    results_df = pd.DataFrame({"PanelID": groups_real, "Prediction": preds_np, "TrueLabel": labels_np})
    results_df["Difference"] = results_df["Prediction"] - results_df["TrueLabel"]
    results_df["AbsDifference"] = results_df["Difference"].abs()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(save_dir, f"wavelet_e{epochs}_seed{run_seed}_{timestamp}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "params": params,
            "epochs": epochs,
            "run_seed": run_seed,
            "metrics": {
                "mae": float(abs_err.mean()),
                "within_1": float((abs_err <= 1).mean() * 100),
                "within_2": float((abs_err <= 2).mean() * 100),
                "within_3": float((abs_err <= 3).mean() * 100),
                "within_4": float((abs_err <= 4).mean() * 100),
            },
        }, path)
        print(f"Saved checkpoint to {path}")

    return model, results_df, preds_np, labels_np, groups_real


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

class _FullInferenceWrapper(nn.Module):
    """Wraps WaveletCNNRegressor with the full preprocessing pipeline so that
    the exported ONNX model accepts a raw camera image directly.

    Expected input
    --------------
    x : float32 tensor of shape [N, 1, H, W], pixel values in [0, 255].
        The default camera resolution is 2464 × 3280.

    Processing steps (must match training exactly)
    -----------------------------------------------
    1. Crop 20 % from each edge  (crop_fraction = 0.4 → 10 % per side in
       half-fraction terms, applied as 20 % total → rows 492:1972, cols 656:2624
       for a 2464 × 3280 image).
    2. Bilinear resize to (img_size × img_size).
    3. Scale to [0, 1] by dividing by 255.
    4. Standardise: (x − 0.5) / 0.5  →  values in [−1, 1].
    """

    def __init__(self, model: nn.Module, img_size: int):
        super().__init__()
        self.model = model
        self.img_size = img_size

    def forward(self, x):
        x = x[:, :, 492:1972, 656:2624]
        x = torch.nn.functional.interpolate(
            x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )
        x = x / 255.0
        x = (x - 0.5) / 0.5
        return self.model(x)


def export_onnx(
    model: nn.Module,
    save_path: str,
    params: dict = None,
    input_height: int = 2464,
    input_width: int = 3280,
) -> str:
    """Export a trained WaveletCNNRegressor to an ONNX file.

    The exported model accepts raw grayscale camera images (pixel values
    0–255) and returns a leveling score.  All preprocessing (cropping,
    resizing, normalisation) is baked into the graph so the caller only
    needs to:

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = img[np.newaxis, np.newaxis, :, :]   # add batch + channel dims
        session = ort.InferenceSession(onnx_path)
        score = float(session.run(None, {"image": img})[0].flat[0])

    Parameters
    ----------
    model : WaveletCNNRegressor
        Trained model returned by ``train_and_evaluate``.
    save_path : str
        Absolute or relative path for the output ``.onnx`` file.
        Use ``os.path.abspath(...)`` to avoid Windows mixed-separator issues.
    params : dict, optional
        Model hyperparameter dict (needs ``"img_size"`` key).
        Defaults to ``DEFAULT_PARAMS``.
    input_height : int
        Expected height of the raw camera image (default 2464).
    input_width : int
        Expected width of the raw camera image (default 3280).

    Returns
    -------
    str
        The resolved absolute path of the saved ONNX file.
    """
    if params is None:
        params = DEFAULT_PARAMS

    img_size = int(params["img_size"])
    save_path = os.path.abspath(save_path)

    model.eval()
    wrapper = _FullInferenceWrapper(model, img_size)
    wrapper.eval()

    dummy = torch.zeros(1, 1, input_height, input_width, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["score"],
        dynamic_axes={"image": {0: "batch_size"}, "score": {0: "batch_size"}},
        dynamo=False,  # legacy exporter avoids .onnx.data sidecar on Windows
    )

    print(f"ONNX model exported to {save_path}")
    return save_path