"""This file contains functions to evaluate the model's performance."""

import numpy as np
import pandas as pd


def c_index(y_true, y_pred):
    """Concordance index between two arrays."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    n = len(y_true)
    if n < 2:
        return np.nan, 0
    concordant = 0.0
    comparable = 0
    for i in range(n - 1):
        dy = y_true[i + 1:] - y_true[i]
        dp = y_pred[i + 1:] - y_pred[i]
        mask = dy != 0
        if not np.any(mask):
            continue
        dy, dp = dy[mask], dp[mask]
        comparable += dy.size
        concordant += np.sum((dy * dp) > 0)
        concordant += 0.5 * np.sum(dp == 0)
    if comparable == 0:
        return np.nan, 0
    return concordant / comparable, comparable


def evaluate_model(preds_np, labels_np, groups):
    """
    Print and return a full evaluation report.

    Parameters
    ----------
    preds_np : 1-D array of model predictions
    labels_np : 1-D array of ground-truth labels
    groups : 1-D array of PanelID values (one per image)

    Returns
    -------
    dict with keys: mae, within_1, within_2, within_3, within_4,
                    c_index_image, c_index_panel, panel_std
    """
    abs_err = np.abs(preds_np - labels_np)

    mae      = float(abs_err.mean())
    within_1 = float((abs_err <= 1).mean() * 100)
    within_2 = float((abs_err <= 2).mean() * 100)
    within_3 = float((abs_err <= 3).mean() * 100)
    within_4 = float((abs_err <= 4).mean() * 100)

    print("\n===== Performance Metrics =====")
    print(f"  MAE:           {mae:.4f}")
    print(f"  Within +/-1:   {within_1:.2f}%")
    print(f"  Within +/-2:   {within_2:.2f}%")
    print(f"  Within +/-3:   {within_3:.2f}%")
    print(f"  Within +/-4:   {within_4:.2f}%")

    # C-index
    c_img, n_img = c_index(labels_np, preds_np)
    panel_df = (
        pd.DataFrame({"PanelID": groups, "TrueLabel": labels_np, "Pred": preds_np})
        .groupby("PanelID", dropna=False)
        .agg(TrueLabel=("TrueLabel", "mean"), Pred=("Pred", "mean"))
        .reset_index()
    )
    c_panel, n_panel = c_index(panel_df["TrueLabel"].values, panel_df["Pred"].values)

    print(f"\n  Image-level C-index: {c_img:.4f}  (pairs: {n_img})")
    print(f"  Panel-level C-index: {c_panel:.4f}  (pairs: {n_panel})")

    # Panel std
    panel_std = (
        pd.DataFrame({"PanelID": groups, "Pred": preds_np, "True": labels_np})
        .assign(Err=lambda d: d["Pred"] - d["True"])
        .groupby("PanelID", dropna=False)
        .agg(
            PredStd=("Pred",  lambda s: s.std(ddof=0)),
            TrueStd=("True",  lambda s: s.std(ddof=0)),
            ErrStd=("Err",   lambda s: s.std(ddof=0)),
        )
        .fillna(0.0)
        .sort_index()
    )
    mean_panel_std = panel_std.mean(numeric_only=True)
    print("\n  Average std across panels:")
    print(mean_panel_std.to_string(float_format=lambda x: f"{x:.4f}"))

    return {
        "mae": mae,
        "within_1": within_1,
        "within_2": within_2,
        "within_3": within_3,
        "within_4": within_4,
        "c_index_image": c_img,
        "c_index_panel": c_panel,
        "panel_std": panel_std,
    }

