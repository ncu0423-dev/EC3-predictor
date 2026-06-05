from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# ユーザーが指定した正規の21特徴量名
CANONICAL_FEATURE_COLUMNS = [
    "KS.EC1.5",
    "h.CLAT.CV75",
    "DPRA.percCysdep",
    "potency_Protein_binding_alerts_for_skin_sensitization_according_to_GHS",
    "DPRA.score",
    "KS.EC3",
    "HOMO Energy",
    "hCLAT.MIT",
    "KS.IC50",
    "DPRA.percLysdep",
    "FM advection sediment",
    "sum_invitro",
    "hCLAT.score",
    "CombDipolPolariz",
    "judge_Protein_binding_alerts_for_skin_sensitization_according_to_GHS",
    "hCLAT.CD86.EC150..ug.ml.",
    "alert_Protein_binding_by_OASIS",
    "OVERALL OH rate constant",
    "LogHL_pred",
    "LUMO Energy",
    "LogKM_pred",
]

# Notebookの学習済みモデルが期待している内部順序
NOTEBOOK_SELECTED_FEATURE_COLUMNS = [
    "OVERALL OH rate constant",
    "FM advection sediment",
    "LUMO Energy",
    "HOMO Energy",
    "CombDipolPolariz",
    "LogHL_pred",
    "LogKM_pred",
    "alert_Protein_binding_by_OASIS",
    "judge_Protein_binding_alerts_for_skin_sensitization_according_to_GHS",
    "potency_Protein_binding_alerts_for_skin_sensitization_according_to_GHS",
    "DPRA.percCysdep",
    "DPRA.percLysdep",
    "DPRA.score",
    "hCLAT.CD86.EC150..ug.ml.",
    "h.CLAT.CV75",
    "hCLAT.MIT",
    "hCLAT.score",
    "KS.EC1.5",
    "KS.EC3",
    "KS.IC50",
    "sum_invitro",
]

MW_COL = "MolWeight"


def load_artifact(path: str | Path) -> dict[str, Any]:
    """保存済みアーティファクトを読み込む。"""
    artifact = joblib.load(path)

    required_top_keys = {
        "artifact_version",
        "full_feature_columns",
        "selected_feature_mask",
        "selected_feature_columns",
        "folds",
    }
    missing = required_top_keys - set(artifact.keys())
    if missing:
        raise ValueError(f"アーティファクトに必要なキーがありません: {sorted(missing)}")

    selected_set = set(artifact["selected_feature_columns"])
    if selected_set != set(CANONICAL_FEATURE_COLUMNS):
        raise ValueError(
            "アーティファクト内の選択特徴量集合が、想定している21特徴量集合と一致しません。"
        )

    return artifact


def y_to_ec3(y: np.ndarray, mol_weight: np.ndarray) -> np.ndarray:
    """Notebookと同じ逆変換: EC3 = MW / (10 ** y)"""
    y = np.asarray(y, dtype=float)
    mol_weight = np.asarray(mol_weight, dtype=float)
    return mol_weight / (10.0 ** y)


def detect_input_mode(
    df: pd.DataFrame, artifact: dict[str, Any]
) -> Literal["full_schema", "selected_only"]:
    """
    入力モード判定。

    - full_schema:
        Notebookで前処理Pipelineがfitされた全列を持つ。最も厳密。
    - selected_only:
        21特徴量 + MolWeight だけを持つ。21特徴量に欠損がない場合のみ、
        選択列部分の標準化を厳密に再現できる。
    """
    full_feature_columns = artifact["full_feature_columns"]
    if set(full_feature_columns).issubset(df.columns):
        return "full_schema"

    minimal_required = set(CANONICAL_FEATURE_COLUMNS + [MW_COL])
    if minimal_required.issubset(df.columns):
        return "selected_only"

    missing = sorted(minimal_required - set(df.columns))
    raise ValueError(
        "入力列が不足しています。少なくとも21特徴量 + MolWeightが必要です。\n"
        + "\n".join(missing)
    )


def validate_selected_only_input(df: pd.DataFrame) -> None:
    """21特徴量のみ入力モードでの厳密性条件を確認する。"""
    missing_cols = [c for c in CANONICAL_FEATURE_COLUMNS + [MW_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError("必要列が不足しています:\n" + "\n".join(missing_cols))

    # 21特徴量に欠損があると、元NotebookのKNNImputer近傍計算を厳密再現できない
    na_counts = df[CANONICAL_FEATURE_COLUMNS].isna().sum()
    bad = na_counts[na_counts > 0]
    if not bad.empty:
        detail = "\n".join([f"{col}: {int(cnt)}" for col, cnt in bad.items()])
        raise ValueError(
            "selected_only モードでは、21特徴量に欠損がある入力は厳密再現できません。\n"
            "欠損がある場合は full_schema（学習時と同じ全特徴量列）を入力してください。\n"
            + detail
        )


def _selected_indices_in_full_schema(artifact: dict[str, Any]) -> list[int]:
    full_cols = list(artifact["full_feature_columns"])
    selected_cols = list(artifact["selected_feature_columns"])
    return [full_cols.index(col) for col in selected_cols]


def transform_for_fold(
    df: pd.DataFrame,
    artifact: dict[str, Any],
    fold_artifact: dict[str, Any],
    mode: Literal["full_schema", "selected_only"],
) -> np.ndarray:
    """
    1 fold 分の前処理を実施する。

    full_schema:
        Notebookと同じ Pipeline.transform -> feature mask.
    selected_only:
        21特徴量が完全に埋まっている場合に限り、
        StandardScaler の selected 列部分だけを使って再現。
    """
    selected_cols = list(artifact["selected_feature_columns"])

    if mode == "full_schema":
        X_full = df[artifact["full_feature_columns"]].copy()
        X_pre = fold_artifact["pipeline"].transform(X_full)
        mask = np.asarray(artifact["selected_feature_mask"], dtype=bool)
        return np.asarray(X_pre)[:, mask]

    validate_selected_only_input(df)

    scaler = fold_artifact["pipeline"].named_steps["standarization"]
    selected_idx = _selected_indices_in_full_schema(artifact)

    mean = np.asarray(scaler.mean_)[selected_idx]
    scale = np.asarray(scaler.scale_)[selected_idx]
    scale = np.where(scale == 0.0, 1.0, scale)

    # モデル内部順序へ並び替え
    X_selected = df[selected_cols].to_numpy(dtype=float)
    X_scaled = (X_selected - mean) / scale
    return X_scaled


def _compute_ad_mask_for_fold(
    x_new_selected: np.ndarray,
    x_train_selected: np.ndarray,
    knn_k: int,
    ad_coverage: float,
) -> np.ndarray:
    """
    Notebookの CALCULATE_KNN_AD_cv / PREDICT_newpred_xgb_cv_model と同じ計算。
    """
    x_dist = cdist(x_train_selected, x_train_selected)
    x_dist_sorted = np.sort(x_dist, axis=1)
    knn_dist = np.mean(x_dist_sorted[:, 1 : knn_k + 1], axis=1)
    knn_dist = np.sort(knn_dist)

    threshold_idx = round(x_train_selected.shape[0] * (1.0 - ad_coverage)) - 1
    threshold_idx = max(0, min(threshold_idx, len(knn_dist) - 1))
    ad_threshold = knn_dist[threshold_idx]

    x_new_dist = cdist(x_new_selected, x_train_selected)
    x_new_dist_sorted = np.sort(x_new_dist, axis=1)
    knn_dist_new = np.mean(x_new_dist_sorted[:, :knn_k], axis=1)

    return knn_dist_new <= ad_threshold


def predict_with_artifact(
    artifact: dict[str, Any],
    input_df: pd.DataFrame,
    compute_ad: bool = True,
) -> pd.DataFrame:
    """
    保存済みアーティファクトを使って推論する。

    出力:
        - pred_y: 5 fold の最大予測値
        - Predict_LLNA_EC3: 5 fold の最小EC3
        - foldごとの pred_y / EC3
        - AD設定が保存されていれば foldごとのAD判定
    """
    mode = detect_input_mode(input_df, artifact)

    if MW_COL not in input_df.columns:
        raise ValueError("MolWeight 列が必要です。")

    mol_weight = input_df[MW_COL].to_numpy(dtype=float)
    fold_pred_y: dict[str, np.ndarray] = {}
    fold_pred_ec3: dict[str, np.ndarray] = {}
    fold_ad: dict[str, np.ndarray] = {}

    ad_config = artifact.get("ad_config", {})
    knn_k = ad_config.get("knn_k")
    ad_coverage = ad_config.get("AD_coverage")

    for fold_name, fold_artifact in artifact["folds"].items():
        x_selected = transform_for_fold(
            df=input_df,
            artifact=artifact,
            fold_artifact=fold_artifact,
            mode=mode,
        )
        model = fold_artifact["model"]
        pred_y = np.asarray(model.predict(x_selected), dtype=float)
        pred_ec3 = y_to_ec3(pred_y, mol_weight)

        fold_pred_y[fold_name] = pred_y
        fold_pred_ec3[fold_name] = pred_ec3

        if compute_ad and knn_k is not None and ad_coverage is not None:
            x_train_selected = np.asarray(fold_artifact["train_selected_X"], dtype=float)
            fold_ad[fold_name] = _compute_ad_mask_for_fold(
                x_new_selected=x_selected,
                x_train_selected=x_train_selected,
                knn_k=int(knn_k),
                ad_coverage=float(ad_coverage),
            )

    pred_y_df = pd.DataFrame(fold_pred_y)
    pred_ec3_df = pd.DataFrame(fold_pred_ec3)

    result = input_df.copy()
    result["inference_schema_mode"] = mode

    for fold_name in pred_y_df.columns:
        result[f"pred_y_{fold_name}"] = pred_y_df[fold_name].values
        result[f"Predict_LLNA_EC3_{fold_name}"] = pred_ec3_df[fold_name].values

    # Notebook後半の出力ロジックと同様に、yは最大、EC3は最小
    result["pred_y"] = pred_y_df.max(axis=1).values
    result["Predict_LLNA_EC3"] = pred_ec3_df.min(axis=1).values

    if fold_ad:
        ad_df = pd.DataFrame(fold_ad)
        for fold_name in ad_df.columns:
            result[f"AD_{fold_name}"] = ad_df[fold_name].values

        result["AD_any_fold"] = ad_df.any(axis=1).values
        result["AD_all_folds"] = ad_df.all(axis=1).values

        mask_y = pred_y_df.where(ad_df)
        mask_ec3 = pred_ec3_df.where(ad_df)

        result["knn_inAD_pred_y"] = mask_y.max(axis=1).values
        result["knn_inAD_Predict_LLNA_EC3"] = mask_ec3.min(axis=1).values

    return result
